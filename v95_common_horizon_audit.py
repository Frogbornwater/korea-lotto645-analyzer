# -*- coding: utf-8 -*-
"""Common-horizon, strict prequential V9.4 vs V5.1 vs uniform audit.

Targets 1002..1251 are forecast using only earlier draws. Offline only:
reads the manually maintained repository CSV; no network access.
"""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from lotto_core import NUMBER_COLS, validate_frame
from v93_multiscale_prequential import WINDOWS, ALPHA, _forecast
from v50_predictive_engine import build_number_features, _fit_from_feature_cache, generate_tickets
from v51_smart_picker import generate_constrained_tickets

TARGET_START = 1002
TARGET_END = 1251
TRAINING_WINDOW = 500
RIDGE_ALPHA = 1.0
MIN_HISTORY = 180
TICKETS = 10
MAX_OVERLAP = 4
BOOT_REPS = 10000
BLOCK = 10
SEED = 20261122

def draw_metrics(p, actual):
    y=np.zeros(45); y[np.asarray(actual,dtype=int)-1]=1
    q=np.clip(np.asarray(p,dtype=float),1e-12,1-1e-12)
    return {
      "brier":float(np.mean((q-y)**2)),
      "logloss":float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))),
      "top6_hits":float(np.sum(y[np.argsort(-q)[:6]]))
    }

def ticket_metrics(tickets, actual):
    hits=[len(set(t)&set(actual)) for t in tickets]
    return {
      "ticket_mean_hits":float(np.mean(hits)),
      "ticket_3plus_rate":float(np.mean(np.asarray(hits)>=3)),
      "batch_3plus":float(any(h>=3 for h in hits)),
      "batch_max_hits":int(max(hits)),
    }

def block_ci(diffs, seed, reps=BOOT_REPS, block=BLOCK):
    a=np.asarray(diffs,dtype=float); n=len(a); rng=np.random.default_rng(seed)
    nb=math.ceil(n/block); means=np.empty(reps)
    for r in range(reps):
        starts=rng.integers(0,n,size=nb)
        sample=np.concatenate([a[(s+np.arange(block))%n] for s in starts])[:n]
        means[r]=sample.mean()
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
    raw=Path("data/lotto645.csv").read_bytes()
    df=validate_frame(pd.read_csv("data/lotto645.csv",encoding="utf-8-sig"))
    draws=df["draw_no"].to_numpy(dtype=int)
    if len(df)!=len(draws) or np.any(np.diff(draws)!=1):
        raise ValueError("CSV必须是连续、有效且无重复的手工历史数据")
    if int(draws[-1]) < TARGET_END:
        raise ValueError(f"需要至少到第{TARGET_END}期，当前仅到{int(draws[-1])}期")
    if int(draws[0]) != 1:
        raise ValueError("本审计要求从第1期开始的连续历史序列")
    matrix=df[NUMBER_COLS].to_numpy(dtype=np.int16)
    # Cache every V5 feature row once. Feature index j uses matrix[:j] only.
    cache={j:build_number_features(matrix[:j]) for j in range(MIN_HISTORY, TARGET_END)}
    uniform=np.full(45,6/45,dtype=float)
    records=[]
    for target in range(TARGET_START,TARGET_END+1):
        idx=int(np.searchsorted(draws,target))
        if idx>=len(draws) or int(draws[idx])!=target or idx<1001:
            raise ValueError(f"目标期{target}不满足严格前瞻条件")
        history=df.iloc[:idx].copy()
        actual=matrix[idx].astype(int).tolist()
        # V9.4: equal-weight fixed windows, strictly pre-target.
        forecasts=[_forecast(matrix[:idx].tolist(),w,ALPHA) for w in WINDOWS]
        p94=np.mean(np.stack(forecasts),axis=0)
        # V5.1's underlying V5.0 ridge model; fit only on past feature/label rows.
        p51,pair,_diag=_fit_from_feature_cache(
          matrix,cache,idx,TRAINING_WINDOW,RIDGE_ALPHA,()
        )
        # Matched deterministic 10-ticket batches and same overlap constraint.
        seed=SEED+target*17
        t94,_,_=generate_constrained_tickets(p94,None,TICKETS,seed,0.0,MAX_OVERLAP)
        t51,_,_=generate_constrained_tickets(p51,pair,TICKETS,seed,0.5,MAX_OVERLAP)
        tr,_,_=generate_constrained_tickets(uniform,None,TICKETS,seed,0.0,MAX_OVERLAP)
        m94=draw_metrics(p94,actual); m51=draw_metrics(p51,actual); mr=draw_metrics(uniform,actual)
        b94=ticket_metrics(t94,actual); b51=ticket_metrics(t51,actual); br=ticket_metrics(tr,actual)
        row={"target_draw":target}
        for name,m,b in (("v94",m94,b94),("v51",m51,b51),("uniform",mr,br)):
            row.update({f"{name}_{k}":v for k,v in m.items()})
            row.update({f"{name}_{k}":v for k,v in b.items()})
        records.append(row)
    detail=pd.DataFrame(records)
    metrics={}
    cols=[c for c in detail.columns if c!="target_draw"]
    for c in cols: metrics[c]=float(detail[c].mean())
    comparisons={}
    endpoints=["brier","logloss","top6_hits","ticket_mean_hits","ticket_3plus_rate","batch_3plus"]
    for baseline in ("v51","uniform"):
        comparisons[f"v94_minus_{baseline}"]={}
        for endpoint in endpoints:
            d=(detail[f"v94_{endpoint}"]-detail[f"{baseline}_{endpoint}"]).to_numpy()
            comparisons[f"v94_minus_{baseline}"][endpoint]={
              "mean_difference":float(d.mean()),
              "paired_circular_moving_block_bootstrap_95ci":block_ci(d,SEED+len(comparisons)*100+len(comparisons[f"v94_minus_{baseline}"]))
            }
    report={
      "version":"9.5.0-common-horizon-prequential-audit",
      "protocol":{
        "target_start":TARGET_START,"target_end":TARGET_END,"target_draws":len(detail),
        "history_cutoff":"Each target forecast uses draws strictly earlier than target draw",
        "v94_windows":list(WINDOWS),"v94_alpha":ALPHA,
        "v51_training_window":TRAINING_WINDOW,"v51_ridge_alpha":RIDGE_ALPHA,
        "tickets_per_model_per_draw":TICKETS,"max_pairwise_ticket_overlap":MAX_OVERLAP,
        "bootstrap":"paired circular moving-block bootstrap, block length 10",
        "random_seed_rule":"same deterministic target-derived seed across models",
        "data_sha256_raw_csv":hashlib.sha256(raw).hexdigest(),
        "latest_draw_in_csv":int(draws[-1]),
      },
      "mean_metrics":metrics,"paired_comparisons":comparisons,
      "interpretation":{
        "predictive_advantage_established":False,
        "note":"Retrospective prequential comparison only. No model is proven to predict a fair independent lottery; bootstrap intervals are descriptive uncertainty, not proof of causal advantage."
      }
    }
    out=Path("artifacts/v95_common_horizon"); out.mkdir(parents=True,exist_ok=True)
    detail.to_csv(out/"v95_detail.csv",index=False)
    (out/"v95_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
