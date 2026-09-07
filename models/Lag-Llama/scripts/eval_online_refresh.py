"""Graph-only online one-step refresh evaluation for Lag-Llama."""

import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from lag_llama.online import CudaGraphFullLagLlamaStep,CudaGraphRollingLagLlamaStep,LagLlamaRollingConfig,RollingLagLlamaEngine,load_pretrained_lag_llama


def summarize(p,t,l):
    p,t,l=map(np.asarray,(p,t,l)); e=p-t
    return {"steps":len(l),"mean_latency_ms":float(l.mean()),"p50_latency_ms":float(np.percentile(l,50)),"mae":float(np.abs(e).mean()),"mse":float(np.square(e).mean())}


@torch.no_grad()
def main(a):
    series=pd.to_numeric(pd.read_csv(a.csv)[a.column],errors="raise").to_numpy(np.float32); refreshes=[int(x) for x in a.refresh_lengths.split(",")]
    model=load_pretrained_lag_llama(a.checkpoint,a.context_length,"cuda"); engine=RollingLagLlamaEngine(model,LagLlamaRollingConfig(context_length=a.context_length)); total=engine.total_length
    if a.start_index<total or a.start_index+a.steps+1>len(series): raise ValueError("invalid evaluation range")
    raw=torch.tensor(series[a.start_index-total:a.start_index],device="cuda").view(1,-1); times=torch.zeros(1,total,6,device="cuda"); engine.full_refresh(raw,times)
    rolling=CudaGraphRollingLagLlamaStep(engine); rolling.capture(); full=CudaGraphFullLagLlamaStep(engine); full.capture(); targets=series[a.start_index+1:a.start_index+a.steps+1]
    methods={}; stored={}; zero=torch.zeros(1,6,device="cuda")
    for refresh in refreshes:
        full.step(raw,times); torch.cuda.synchronize(); window=raw.clone(); time_window=times.clone(); preds=[]; lats=[]; calls=0
        for i in range(a.steps):
            value=torch.tensor([[series[a.start_index+i]]],device="cuda"); window=torch.cat((window[:,1:],value),-1); time_window=torch.cat((time_window[:,1:],zero.unsqueeze(1)),1); use=refresh>0 and (i+1)%refresh==0
            begin=time.perf_counter(); pred=full.step(window,time_window) if use else rolling.step(value,zero); torch.cuda.synchronize(); lats.append((time.perf_counter()-begin)*1000); preds.append(float(pred.item())); calls+=int(use)
        key=str(refresh); stored[key]=np.asarray(preds); methods[key]=summarize(stored[key],targets,lats); methods[key].update({"refresh_length":refresh,"full_graph_replays":calls,"rolling_graph_replays":a.steps-calls}); print(f"K={refresh} latency={methods[key]['mean_latency_ms']:.3f} MAE={methods[key]['mae']:.6f}")
    base=stored["1"]; base_mae=methods["1"]["mae"]
    for k,p in stored.items(): methods[k]["prediction_gap_mae_vs_full_k1"]=float(np.abs(p-base).mean()); methods[k]["forecast_mae_delta_vs_full_k1"]=methods[k]["mae"]-base_mae
    out={"model":"Lag-Llama","execution":"cuda_graph_only","zero_time_features":True,"dataset":{"csv":os.path.abspath(a.csv),"column":a.column,"start_index":a.start_index},"context_length":a.context_length,"max_lag":engine.max_lag,"prediction_length":1,"methods":methods}; os.makedirs(os.path.dirname(os.path.abspath(a.output)),exist_ok=True); json.dump(out,open(a.output,"w"),indent=2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint",required=True); p.add_argument("--csv",required=True); p.add_argument("--column",required=True); p.add_argument("--start-index",type=int,required=True); p.add_argument("--steps",type=int,default=128); p.add_argument("--context-length",type=int,default=2048); p.add_argument("--refresh-lengths",default="1,4,16,0"); p.add_argument("--output",required=True); main(p.parse_args())
