# -*- coding: utf-8 -*-
"""Strict prequential scalar temperature calibration; offline historical data only."""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lotto_core import NUMBER_COLS, validate_frame
from v50_backtest import _probability_scores, _v4_long_frequency_probabilities, _bootstrap_mean_ci
from v50_predictive_engine import _fit_from_feature_cache, _calibrate_to_six, build_number_features

TRAINING_WINDOW=500
MIN_HISTORY=180
CALIBRATION_DRAWS=50
EVALUATION_DRAWS=50
RIDGE_ALPHA=1.0
# Wider search: the previous selected value was exactly the upper bound (2.0).
# A boundary optimum is explicitly flagged and must not be treated as settled.
TEMPERATURE_GRID=tuple(float(x) for x in np.round(np.arange(0.25,4.0001,0.025),3))
BOOTSTRAP_REPS=2000
SEED=5527

def _logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9)
    return np.log(p/(1-p))

def transform(p,temperature):
    if not np.isfinite(temperature) or temperature<=0: raise ValueError("temperature必须为正有限数值。")
    return _calibrate_to_six(_logit(p)/float(temperature))

def mean_logloss(p,y):
    p=np.clip(np.asarray(p,dtype=float),1e-9,1-1e-9); y=np.asarray(y,dtype=float)
    if p.shape!=y.shape or p.ndim!=2 or p.shape[1]!=45: raise ValueError("概率和结果必须是相同形状的N×45矩阵。")
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))

def select_temperature(probabilities,outcomes):
    p=np.asarray(probabilities,dtype=float); y=np.asarray(outcomes,dtype=float)
    if p.ndim!=2 or p.shape[1]!=45 or p.shape!=y.shape or len(p)<10: raise ValueError("校准数据必须至少包含10期N×45矩阵。")
    if not np.all(np.isfinite(p)) or np.any(p<=0) or np.any(p>=1): raise ValueError("校准概率必须是(0,1)内的有限数值。")
    if np.any((y!=0)&(y!=1)) or not np.all(y.sum(axis=1)==6): raise ValueError("每期结果必须恰好包含6个1。")
    scored=[]
    for t in TEMPERATURE_GRID:
        adjusted=np.vstack([transform(row,t) for row in p])
        scored.append((mean_logloss(adjusted,y),t))
    loss,t=min(scored,key=lambda z:(z[0],abs(z[1]-1),z[1]))
    boundary = bool(np.isclose(t,TEMPERATURE_GRID[0]) or np.isclose(t,TEMPERATURE_GRID[-1]))
    return {
        "temperature":float(t),
        "calibration_log_loss":float(loss),
        "raw_calibration_log_loss":mean_logloss(p,y),
        "grid":list(TEMPERATURE_GRID),
        "grid_min":float(TEMPERATURE_GRID[0]),
        "grid_max":float(TEMPERATURE_GRID[-1]),
        "grid_boundary_optimum":boundary,
        "grid_status":"boundary_optimum_requires_wider_search" if boundary else "interior_grid_optimum",
    }

def outcome_vector(nums):
    y=np.zeros(45); y[np.asarray(nums,dtype=int)-1]=1
    return y

def reliability(p,y,bins=10):
    p=np.asarray(p).reshape(-1); y=np.asarray(y).reshape(-1)
    groups=np.array_split(np.argsort(p,kind="mergesort"),bins); rows=[]; ece=0.0
    for i,g in enumerate(groups,1):
        mp=float(p[g].mean()); obs=float(y[g].mean()); ece+=len(g)/len(p)*abs(obs-mp)
        rows.append({"bin":i,"count":len(g),"mean_predicted_probability":mp,"observed_frequency":obs,"gap":obs-mp})
    return {"bins":rows,"equal_count_ECE":float(ece)}

def run_experiment(frame):
    clean=validate_frame(frame).reset_index(drop=True)
    draws=clean.draw_no.to_numpy(dtype=int); matrix=clean[NUMBER_COLS].to_numpy(dtype=np.int16,copy=True)
    needed=MIN_HISTORY+CALIBRATION_DRAWS+EVALUATION_DRAWS
    if len(clean)<needed: raise ValueError(f"至少需要{needed}期连续历史数据。")
    if np.any(np.diff(draws)!=1): raise ValueError("期号不连续，拒绝运行温度校准实验。")
    cal_start=len(clean)-CALIBRATION_DRAWS-EVALUATION_DRAWS; eval_start=cal_start+CALIBRATION_DRAWS
    cache={i:build_number_features(matrix[:i]) for i in range(MIN_HISTORY,len(matrix))}
    cal_p=[]; cal_y=[]
    for i in range(cal_start,eval_start):
        p,_,_=_fit_from_feature_cache(matrix,cache,i,TRAINING_WINDOW,RIDGE_ALPHA)
        cal_p.append(p); cal_y.append(outcome_vector(matrix[i]))
    cal_p=np.vstack(cal_p); cal_y=np.vstack(cal_y)
    selected=select_temperature(cal_p,cal_y); temperature=selected["temperature"]
    raw_rows=[]; adjusted_rows=[]; v4_rows=[]; uniform_rows=[]; y_rows=[]; per=[]
    for i in range(eval_start,len(clean)):
        raw,_,_=_fit_from_feature_cache(matrix,cache,i,TRAINING_WINDOW,RIDGE_ALPHA)
        adjusted=transform(raw,temperature)
        v4=_v4_long_frequency_probabilities(matrix[max(0,i-TRAINING_WINDOW):i])
        uniform=np.full(45,6/45); y=outcome_vector(matrix[i]); actual=set(matrix[i].tolist())
        rb,rl=_probability_scores(raw,actual); cb,cl=_probability_scores(adjusted,actual)
        vb,vl=_probability_scores(v4,actual); ub,ul=_probability_scores(uniform,actual)
        per.append({"draw":int(draws[i]),"raw_brier":rb,"calibrated_brier":cb,"v4_brier":vb,"uniform_brier":ub,
                    "raw_log_loss":rl,"calibrated_log_loss":cl,"v4_log_loss":vl,"uniform_log_loss":ul,
                    "raw_probability_sum":float(raw.sum()),"calibrated_probability_sum":float(adjusted.sum())})
        raw_rows.append(raw); adjusted_rows.append(adjusted); v4_rows.append(v4); uniform_rows.append(uniform); y_rows.append(y)
    raw=np.vstack(raw_rows); adjusted=np.vstack(adjusted_rows); v4=np.vstack(v4_rows); uniform=np.vstack(uniform_rows); y=np.vstack(y_rows)
    def scores(p): return {"brier":float(np.mean((p-y)**2)),"log_loss":mean_logloss(p,y),"reliability":reliability(p,y)}
    per_df=pd.DataFrame(per); comparisons={}
    for metric in ("brier","log_loss"):
        d_raw=per_df[f"raw_{metric}"].to_numpy()-per_df[f"calibrated_{metric}"].to_numpy()
        d_v4=per_df[f"v4_{metric}"].to_numpy()-per_df[f"calibrated_{metric}"].to_numpy()
        d_uni=per_df[f"uniform_{metric}"].to_numpy()-per_df[f"calibrated_{metric}"].to_numpy()
        comparisons[metric]={"improvement_vs_raw":float(d_raw.mean()),"vs_raw_95ci":list(_bootstrap_mean_ci(d_raw,BOOTSTRAP_REPS,SEED)),
          "improvement_vs_v4":float(d_v4.mean()),"vs_v4_95ci":list(_bootstrap_mean_ci(d_v4,BOOTSTRAP_REPS,SEED+1)),
          "improvement_vs_uniform":float(d_uni.mean()),"vs_uniform_95ci":list(_bootstrap_mean_ci(d_uni,BOOTSTRAP_REPS,SEED+2))}
    return {"experiment":"V5.5 strictly prequential scalar temperature calibration","data_first_draw":int(draws[0]),"data_last_draw":int(draws[-1]),
      "calibration_period":[int(draws[cal_start]),int(draws[eval_start-1])],"evaluation_period":[int(draws[eval_start]),int(draws[-1])],
      "calibration_draws":CALIBRATION_DRAWS,"evaluation_draws":EVALUATION_DRAWS,"training_window":TRAINING_WINDOW,"ridge_alpha":RIDGE_ALPHA,
      "selected_temperature":selected,"evaluation_scores":{"raw_v5":scores(raw),"calibrated_v5":scores(adjusted),"v4_frozen":scores(v4),"uniform":scores(uniform)},
      "paired_draw_comparisons":comparisons,"per_draw":per,"data_fingerprint_sha256":hashlib.sha256(clean.to_csv(index=False).encode("utf-8")).hexdigest(),
      "interpretation":"Temperature selected only on earlier rolling out-of-fold predictions, then frozen for later evaluation. Retrospective only; may overlap earlier research.",
      "promotion_policy":"Do not promote calibration based on this single retrospective split."}

def main():
    print(json.dumps(run_experiment(pd.read_csv(ROOT/"data"/"lotto645.csv")),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
