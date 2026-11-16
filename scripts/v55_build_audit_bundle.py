# -*- coding: utf-8 -*-
"""Build reproducible V5.5 audit reports and a frozen calibration lock.

All inputs come from the manually maintained repository CSV. The lock is
calibrated on the latest 50 prequential historical predictions and is intended
ONLY for draws strictly after its recorded base draw. It is not applied to
historical evaluation rows.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"artifacts"/"v55"
sys.path.insert(0,str(ROOT))
from lotto_core import NUMBER_COLS, validate_frame
from v50_predictive_engine import build_number_features, _fit_from_feature_cache
from scripts.v55_prequential_temperature_calibration import select_temperature
from scripts.v55_prequential_temperature_calibration import outcome_vector
from scripts.v55_prequential_temperature_calibration import RIDGE_ALPHA, TRAINING_WINDOW, MIN_HISTORY
from scripts.v55_prequential_temperature_calibration import CALIBRATION_DRAWS
from scripts.v55_prequential_temperature_calibration import TEMPERATURE_GRID
from scripts.v55_prequential_temperature_calibration import transform
from scripts.v55_prequential_temperature_calibration import run_experiment
from scripts.v55_ridge_calibration_diagnostics import run_diagnostics

def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()

def build_forward_lock(frame: pd.DataFrame) -> dict:
    clean=validate_frame(frame).reset_index(drop=True)
    draws=clean["draw_no"].to_numpy(dtype=int)
    matrix=clean[NUMBER_COLS].to_numpy(dtype=np.int16,copy=True)
    if len(clean)<MIN_HISTORY+CALIBRATION_DRAWS:
        raise ValueError("历史数据不足，无法生成温度校准锁定。")
    if np.any(np.diff(draws)!=1):
        raise ValueError("期号不连续，拒绝生成前瞻锁定。")
    start=len(clean)-CALIBRATION_DRAWS
    cache={i:build_number_features(matrix[:i]) for i in range(MIN_HISTORY,len(matrix))}
    probabilities=[]; outcomes=[]
    for i in range(start,len(clean)):
        p,_,_=_fit_from_feature_cache(matrix,cache,i,TRAINING_WINDOW,RIDGE_ALPHA)
        probabilities.append(p); outcomes.append(outcome_vector(matrix[i]))
    selected=select_temperature(np.vstack(probabilities),np.vstack(outcomes))
    canonical=clean.to_csv(index=False).encode("utf-8")
    return {
      "lock_version":"v55-temperature-lock-1",
      "method":"scalar temperature scaling on 50 strictly prequential historical forecasts",
      "temperature":selected["temperature"],
      "calibration_log_loss":selected["calibration_log_loss"],
      "raw_calibration_log_loss":selected["raw_calibration_log_loss"],
      "calibration_draws":CALIBRATION_DRAWS,
      "calibration_first_draw":int(draws[start]),
      "calibration_last_draw":int(draws[-1]),
      "base_last_draw":int(draws[-1]),
      "base_row_count":int(len(clean)),
      "history_sha256":sha256_bytes(canonical),
      "training_window":TRAINING_WINDOW,
      "ridge_alpha":RIDGE_ALPHA,
      "temperature_grid":list(TEMPERATURE_GRID),
      "frozen":True,
      "activation_rule":"Apply only to target draws strictly greater than base_last_draw; never refit temperature during forward evaluation.",
      "promotion_policy":"Research-only. Do not change production predictions until multiple prospective evaluations support improvement."
    }

def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\\n",encoding="utf-8")

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    history=pd.read_csv(ROOT/"data"/"lotto645.csv")
    ridge_path=OUT/"ridge_reliability_report.json"
    temp_path=OUT/"temperature_retrospective_report.json"
    ridge=json.loads(ridge_path.read_text(encoding="utf-8")) if ridge_path.exists() else run_diagnostics(history)
    temp=json.loads(temp_path.read_text(encoding="utf-8")) if temp_path.exists() else run_experiment(history)
    lock=build_forward_lock(history)
    write_json(OUT/"ridge_reliability_report.json",ridge)
    write_json(OUT/"temperature_retrospective_report.json",temp)
    write_json(OUT/"forward_temperature_lock.json",lock)
    pd.DataFrame(ridge["cases"]).to_csv(OUT/"ridge_grid_summary.csv",index=False)
    pd.DataFrame(temp["per_draw"]).to_csv(OUT/"temperature_evaluation_per_draw.csv",index=False)
    reliability=[]
    for model,report in temp["evaluation_scores"].items():
        for row in report["reliability"]["bins"]:
            reliability.append({"model":model,**row})
    pd.DataFrame(reliability).to_csv(OUT/"temperature_reliability_bins.csv",index=False)
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!="manifest.json")
    manifest={
      "report_bundle":"V5.5 Ridge, reliability, temperature calibration and forward lock",
      "data_first_draw":int(history.draw_no.min()),"data_last_draw":int(history.draw_no.max()),
      "data_rows":int(len(history)),
      "data_sha256":sha256_bytes(history.to_csv(index=False).encode("utf-8")),
      "files":{p.name:{"sha256":sha256_bytes(p.read_bytes()),"bytes":p.stat().st_size} for p in files},
      "limitations":[
        "The retrospective 50-draw evaluation may overlap earlier project work and is not independent prospective evidence.",
        "The forward temperature lock is calibrated on the latest 50 historical prequential predictions and applies only after its base_last_draw.",
        "No production model settings are changed by generating this report.",
        "All lottery history is manually supplied; no official lottery website, API, scraping or automatic synchronization is accessed."
      ]
    }
    write_json(OUT/"manifest.json",manifest)
    print(json.dumps({"output_dir":str(OUT.relative_to(ROOT)),"manifest":manifest},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
