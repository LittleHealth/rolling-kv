"""Graph-only online refresh evaluation for Timer-XL."""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from timer_xl_online import CudaGraphFullTimerXLStep, CudaGraphRollingTimerXLStep, RollingTimerXLEngine, TimerXLRollingConfig, load_pretrained_timer_xl


def summarize(p,t,l):
    p,t,l=map(np.asarray,(p,t,l)); e=p-t
    return {"steps":len(l),"mean_latency_ms":float(l.mean()),"p50_latency_ms":float(np.percentile(l,50)),"mae":float(np.abs(e).mean()),"mse":float(np.square(e).mean())}


@torch.no_grad()
def main(a):
    series=pd.to_numeric(pd.read_csv(a.csv)[a.column],errors="raise").to_numpy(np.float32); refreshes=[int(x) for x in a.refresh_lengths.split(",")]
    model=load_pretrained_timer_xl(a.checkpoint,"cuda"); patch=96; required=a.start_index+a.steps*patch+patch
    if a.start_index<a.context_length or required>len(series): raise ValueError(f"need {required}, have {len(series)}")
    initial=torch.tensor(series[a.start_index-a.context_length:a.start_index],device="cuda").view(1,-1,1)
    engine=RollingTimerXLEngine(model,TimerXLRollingConfig(context_length=a.context_length)); engine.full_refresh(initial)
    rolling=CudaGraphRollingTimerXLStep(engine); rolling.capture(); full=CudaGraphFullTimerXLStep(engine); full.capture()
    targets=np.stack([series[s+patch:s+2*patch] for s in range(a.start_index,a.start_index+a.steps*patch,patch)])
    methods={}; stored={}
    for refresh in refreshes:
        full.step(initial); torch.cuda.synchronize(); window=initial.clone(); preds=[]; lats=[]; calls=0
        for i,start in enumerate(range(a.start_index,a.start_index+a.steps*patch,patch)):
            patch_x=torch.tensor(series[start:start+patch],device="cuda").view(1,patch,1); window=torch.cat((window[:,patch:],patch_x),1); use=refresh>0 and (i+1)%refresh==0
            begin=time.perf_counter(); pred=full.step(window) if use else rolling.step(patch_x); torch.cuda.synchronize(); lats.append((time.perf_counter()-begin)*1000); preds.append(pred[0].cpu().numpy().copy()); calls+=int(use)
        key=str(refresh); stored[key]=np.stack(preds); methods[key]=summarize(stored[key],targets,lats); methods[key].update({"refresh_length":refresh,"full_graph_replays":calls,"rolling_graph_replays":a.steps-calls}); print(f"K={refresh} latency={methods[key]['mean_latency_ms']:.3f} MAE={methods[key]['mae']:.6f}")
    for threshold in [float(x) for x in a.adaptive_thresholds.split(",") if x]:
        full.step(initial); torch.cuda.synchronize(); window=initial.clone(); window_np=initial[:, :, 0].cpu().numpy().copy(); ref_mean=float(window_np.mean()); ref_std=float(window_np.std())+1e-6; preds=[]; lats=[]; calls=0; since=0
        for start in range(a.start_index,a.start_index+a.steps*patch,patch):
            new_np=series[start:start+patch]; patch_x=torch.tensor(new_np,device="cuda").view(1,patch,1); window=torch.cat((window[:,patch:],patch_x),1); window_np=np.concatenate((window_np[:,patch:],new_np[None]),1); since+=1; drift=max(abs(float(window_np.mean())-ref_mean)/ref_std,abs((float(window_np.std())+1e-6)-ref_std)/ref_std); use=since>=a.adaptive_max_refresh or (since>=a.adaptive_min_refresh and drift>=threshold)
            begin=time.perf_counter(); pred=full.step(window) if use else rolling.step(patch_x); torch.cuda.synchronize(); lats.append((time.perf_counter()-begin)*1000); preds.append(pred[0].cpu().numpy().copy()); calls+=int(use)
            if use: ref_mean=float(window_np.mean()); ref_std=float(window_np.std())+1e-6; since=0
        key=f"adaptive_{threshold:g}"; stored[key]=np.stack(preds); methods[key]=summarize(stored[key],targets,lats); methods[key].update({"refresh_policy":"normalization_drift","drift_threshold":threshold,"min_refresh_length":a.adaptive_min_refresh,"max_refresh_length":a.adaptive_max_refresh,"full_graph_replays":calls,"rolling_graph_replays":a.steps-calls}); print(f"adaptive={threshold:g} full={calls} latency={methods[key]['mean_latency_ms']:.3f} MAE={methods[key]['mae']:.6f}")
    base=stored["1"]; base_mae=methods["1"]["mae"]
    for k,p in stored.items(): methods[k]["prediction_gap_mae_vs_full_k1"]=float(np.abs(p-base).mean()); methods[k]["forecast_mae_delta_vs_full_k1"]=methods[k]["mae"]-base_mae
    out={"model":"Timer-XL-67M","execution":"cuda_graph_only","dataset":{"csv":os.path.abspath(a.csv),"column":a.column,"start_index":a.start_index},"context_length":a.context_length,"context_tokens":a.context_length//patch,"patch_length":patch,"prediction_length":patch,"methods":methods}; os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True); json.dump(out,open(a.output,"w"),indent=2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True); p.add_argument("--csv",required=True); p.add_argument("--column",required=True); p.add_argument("--start-index",type=int,required=True); p.add_argument("--steps",type=int,default=32); p.add_argument("--context-length",type=int,default=12288); p.add_argument("--refresh-lengths",default="1,4,16,0"); p.add_argument("--adaptive-thresholds",default=""); p.add_argument("--adaptive-min-refresh",type=int,default=2); p.add_argument("--adaptive-max-refresh",type=int,default=8); p.add_argument("--output",required=True); main(p.parse_args())
