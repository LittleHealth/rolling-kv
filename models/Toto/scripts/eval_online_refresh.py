"""Graph-only online refresh evaluation for Toto 2.0."""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "toto2"))
sys.path.insert(0, os.path.join(ROOT, "dd_unit_scaling"))

from toto2 import Toto2Model
from toto2.online_rolling import CudaGraphFullToto2Step, CudaGraphRollingToto2Step, RollingToto2Engine, Toto2RollingConfig


def metrics(predictions, targets, latencies):
    prediction, target, latency = map(np.asarray, (predictions, targets, latencies))
    error = prediction - target
    return {
        "steps": len(latency),
        "mean_latency_ms": float(latency.mean()),
        "p50_latency_ms": float(np.percentile(latency, 50)),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
    }


@torch.no_grad()
def main(args):
    refreshes = [int(x) for x in args.refresh_lengths.split(",")]
    frame = pd.read_csv(args.csv)
    series = pd.to_numeric(frame[args.column], errors="raise").to_numpy(np.float32)
    model = Toto2Model.from_pretrained(args.checkpoint, map_location="cpu").cuda().eval()
    patch = model.config.patch_size
    cfg = Toto2RollingConfig(context_length=args.context_length, horizon=args.horizon)
    required = args.start_index + args.steps * patch + args.horizon
    if args.start_index < args.context_length or required > len(series):
        raise ValueError(f"invalid range: need {required}, series has {len(series)}")
    initial = torch.tensor(series[args.start_index-args.context_length:args.start_index], device="cuda").view(1, 1, -1)
    engine = RollingToto2Engine(model, cfg); engine.full_refresh(initial)
    rolling = CudaGraphRollingToto2Step(engine); rolling.capture()
    full = CudaGraphFullToto2Step(engine); full.capture()
    targets = np.stack([series[s+patch:s+patch+args.horizon] for s in range(args.start_index, args.start_index+args.steps*patch, patch)])
    methods, stored = {}, {}
    for refresh in refreshes:
        full.step(initial); torch.cuda.synchronize()
        window = initial.clone(); predictions=[]; latencies=[]; full_calls=0
        for index, start in enumerate(range(args.start_index, args.start_index+args.steps*patch, patch)):
            new_patch = torch.tensor(series[start:start+patch], device="cuda").view(1,1,-1)
            window = torch.cat((window[..., patch:], new_patch), -1)
            use_full = refresh > 0 and (index + 1) % refresh == 0
            begin=time.perf_counter(); prediction = full.step(window) if use_full else rolling.step(new_patch); torch.cuda.synchronize()
            latencies.append((time.perf_counter()-begin)*1000); predictions.append(prediction[0,0].cpu().numpy().copy()); full_calls += int(use_full)
        key=str(refresh); stored[key]=np.stack(predictions); methods[key]=metrics(stored[key], targets, latencies)
        methods[key].update({"refresh_length":refresh,"full_graph_replays":full_calls,"rolling_graph_replays":args.steps-full_calls})
        print(f"K={refresh} latency={methods[key]['mean_latency_ms']:.3f} MAE={methods[key]['mae']:.6f}")
    for threshold in [float(x) for x in args.adaptive_thresholds.split(",") if x]:
        full.step(initial); torch.cuda.synchronize()
        window=initial.clone(); window_np=initial[0,0].cpu().numpy().copy()
        ref_mean=float(window_np.mean()); ref_std=float(window_np.std())+1e-6
        predictions=[]; latencies=[]; full_calls=0; since_refresh=0
        for start in range(args.start_index,args.start_index+args.steps*patch,patch):
            new_np=series[start:start+patch]
            new_patch=torch.tensor(new_np,device="cuda").view(1,1,-1)
            window=torch.cat((window[...,patch:],new_patch),-1)
            window_np=np.concatenate((window_np[patch:],new_np))
            since_refresh+=1
            drift=max(abs(float(window_np.mean())-ref_mean)/ref_std,abs((float(window_np.std())+1e-6)-ref_std)/ref_std)
            use_full=since_refresh>=args.adaptive_max_refresh or (since_refresh>=args.adaptive_min_refresh and drift>=threshold)
            begin=time.perf_counter(); prediction=full.step(window) if use_full else rolling.step(new_patch); torch.cuda.synchronize()
            latencies.append((time.perf_counter()-begin)*1000); predictions.append(prediction[0,0].cpu().numpy().copy()); full_calls+=int(use_full)
            if use_full:
                ref_mean=float(window_np.mean()); ref_std=float(window_np.std())+1e-6; since_refresh=0
        key=f"adaptive_{threshold:g}"; stored[key]=np.stack(predictions); methods[key]=metrics(stored[key],targets,latencies)
        methods[key].update({"refresh_policy":"normalization_drift","drift_threshold":threshold,"min_refresh_length":args.adaptive_min_refresh,"max_refresh_length":args.adaptive_max_refresh,"full_graph_replays":full_calls,"rolling_graph_replays":args.steps-full_calls})
        print(f"adaptive={threshold:g} full={full_calls} latency={methods[key]['mean_latency_ms']:.3f} MAE={methods[key]['mae']:.6f}")
    baseline=stored["1"]; baseline_mae=methods["1"]["mae"]
    for key,prediction in stored.items():
        methods[key]["prediction_gap_mae_vs_full_k1"]=float(np.abs(prediction-baseline).mean())
        methods[key]["forecast_mae_delta_vs_full_k1"]=methods[key]["mae"]-baseline_mae
    output={"model":"Toto-2.0","execution":"cuda_graph_only","dataset":{"csv":os.path.abspath(args.csv),"column":args.column,"start_index":args.start_index},"context_length":args.context_length,"context_tokens":args.context_length//patch,"patch_length":patch,"prediction_length":args.horizon,"methods":methods}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)),exist_ok=True)
    with open(args.output,"w") as handle: json.dump(output,handle,indent=2)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True); p.add_argument("--csv",required=True); p.add_argument("--column",required=True); p.add_argument("--start-index",type=int,required=True); p.add_argument("--steps",type=int,default=64); p.add_argument("--context-length",type=int,default=8192); p.add_argument("--horizon",type=int,default=32); p.add_argument("--refresh-lengths",default="1,4,16,0"); p.add_argument("--adaptive-thresholds",default=""); p.add_argument("--adaptive-min-refresh",type=int,default=4); p.add_argument("--adaptive-max-refresh",type=int,default=32); p.add_argument("--output",required=True); main(p.parse_args())
