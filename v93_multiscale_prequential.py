# -*- coding: utf-8 -*-
"""V9.3 strict prequential multi-timescale frequency-signal research."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, random
from pathlib import Path
from typing import Any
import numpy as np

WINDOWS = (20, 30, 50, 100, 200, 300, 500, 800, 1000)
ALPHA = 80.0
UNIFORM_P = 6.0 / 45.0
BOOTSTRAP_REPS = 10000


def load_draws(path: str | Path) -> tuple[list[int], list[list[int]], str]:
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("开奖数据为空")
    draws, numbers = [], []
    for row in rows:
        draw = int(row["draw_no"])
        vals = [int(row[f"n{i}"]) for i in range(1, 7)]
        if len(set(vals)) != 6 or any(x < 1 or x > 45 for x in vals):
            raise ValueError(f"第{draw}期号码无效")
        draws.append(draw); numbers.append(sorted(vals))
    if draws != sorted(set(draws)):
        raise ValueError("期号必须唯一且严格升序")
    return draws, numbers, hashlib.sha256(raw).hexdigest()


def _forecast(history: list[list[int]], window: int, alpha: float = ALPHA) -> np.ndarray:
    if len(history) < window:
        raise ValueError("训练历史不足指定窗口")
    counts = np.zeros(45, dtype=float)
    for row in history[-window:]:
        counts[np.asarray(row, dtype=int) - 1] += 1.0
    p = (counts + alpha * UNIFORM_P) / (window + alpha)
    # Marginal inclusion probabilities sum to six in exact arithmetic.
    return p


def _draw_scores(p: np.ndarray, actual: list[int]) -> dict[str, float]:
    y = np.zeros(45, dtype=float); y[np.asarray(actual) - 1] = 1.0
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return {
        "brier": float(np.mean((p-y)**2)),
        "logloss": float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
        "top6_hits": float(np.sum(y[np.argsort(-p)[:6]])),
    }


def _block_bootstrap_ci(diffs: list[float], seed: int, reps: int = BOOTSTRAP_REPS, block: int = 10) -> list[float]:
    n = len(diffs)
    if n == 0: raise ValueError("没有可评估目标期")
    arr = np.asarray(diffs, dtype=float); rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    means = np.empty(reps)
    for r in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.concatenate([arr[(s + np.arange(block)) % n] for s in starts])[:n]
        means[r] = sample.mean()
    return [float(np.quantile(means,.025)), float(np.quantile(means,.975))]


def _block_sign_flip_p(diffs: list[float], seed: int, reps: int = BOOTSTRAP_REPS, block: int = 10) -> float:
    arr=np.asarray(diffs,dtype=float); obs=float(arr.mean()); rng=np.random.default_rng(seed)
    # Apply one random sign per contiguous block, preserving within-block dependence.
    chunks=[arr[i:i+block] for i in range(0,len(arr),block)]
    extreme=0
    for _ in range(reps):
        signs=rng.choice(np.array([-1.0,1.0]),size=len(chunks))
        value=float(np.mean(np.concatenate([chunk*sign for chunk,sign in zip(chunks,signs)])))
        if value <= obs: extreme += 1
    return (extreme+1)/(reps+1)


def _holm(pvals: dict[str,float]) -> dict[str,float]:
    ordered=sorted(pvals.items(),key=lambda kv:kv[1]); m=len(ordered); out={}; running=0.0
    for i,(name,p) in enumerate(ordered):
        running=max(running,min(1.0,(m-i)*p)); out[name]=running
    return out


def analyze_v93(draws: list[int], numbers: list[list[int]], bootstrap_reps: int = BOOTSTRAP_REPS) -> dict[str,Any]:
    if len(draws)!=len(numbers) or len(draws)<max(WINDOWS)+1:
        raise ValueError("数据长度不足；至少需要1001期有效开奖")
    if draws != sorted(set(draws)): raise ValueError("期号必须唯一且严格升序")
    if bootstrap_reps < 1000: raise ValueError("bootstrap次数至少1000")
    for row in numbers:
        if len(row)!=6 or len(set(row))!=6 or any(int(x)<1 or int(x)>45 for x in row):
            raise ValueError("每期必须包含六个不重复的1至45号码")
    names=[f"window_{w}" for w in WINDOWS]+["equal_multiscale"]
    rows={name:[] for name in names}; uniform=[]
    # Strict prequential: for target index i, only rows[:i] are visible.
    for i in range(max(WINDOWS),len(draws)):
        history=numbers[:i]; actual=numbers[i]
        forecasts={f"window_{w}":_forecast(history,w) for w in WINDOWS}
        forecasts["equal_multiscale"]=np.mean(np.stack(list(forecasts.values())),axis=0)
        uniform.append(_draw_scores(np.full(45,UNIFORM_P),actual))
        for name,p in forecasts.items(): rows[name].append(_draw_scores(p,actual))
    metrics={}; primary_p={}
    for name,vals in rows.items():
        bdiff=[v["brier"]-u["brier"] for v,u in zip(vals,uniform)]
        ldiff=[v["logloss"]-u["logloss"] for v,u in zip(vals,uniform)]
        hdiff=[v["top6_hits"]-u["top6_hits"] for v,u in zip(vals,uniform)]
        primary_p[name]=_block_sign_flip_p(bdiff,seed=93000+len(primary_p),reps=bootstrap_reps,block=10)
        metrics[name]={
            "draws":len(vals),
            "mean_brier":float(np.mean([v["brier"] for v in vals])),
            "mean_logloss":float(np.mean([v["logloss"] for v in vals])),
            "mean_top6_hits":float(np.mean([v["top6_hits"] for v in vals])),
            "brier_difference_vs_uniform":float(np.mean(bdiff)),
            "brier_difference_95ci_block10":_block_bootstrap_ci(bdiff,93001+len(metrics),bootstrap_reps),
            "logloss_difference_vs_uniform":float(np.mean(ldiff)),
            "logloss_difference_95ci_block10":_block_bootstrap_ci(ldiff,94001+len(metrics),bootstrap_reps),
            "top6_hits_difference_vs_uniform":float(np.mean(hdiff)),
            "top6_hits_difference_95ci_block10":_block_bootstrap_ci(hdiff,95001+len(metrics),bootstrap_reps),
            "brier_one_sided_block_signflip_p":primary_p[name],
        }
    adjusted=_holm(primary_p)
    for name in names: metrics[name]["brier_holm_adjusted_p"]=adjusted[name]
    return {
        "version":"9.3.0-multiscale-prequential",
        "protocol":{"target_start_draw":draws[max(WINDOWS)],"target_end_draw":draws[-1],
          "target_draws":len(draws)-max(WINDOWS),"training_windows":list(WINDOWS),
          "dirichlet_smoothing_alpha":ALPHA,"primary_endpoint":"per-draw 45-dimensional Brier score",
          "primary_family":"9 fixed windows plus equal-weight multiscale ensemble",
          "bootstrap":"paired circular moving-block bootstrap, block length 10; one-sided block sign-flip randomization, contiguous block length 10",
          "prediction_timing":"strictly before each target draw; no target/future leakage"},
        "uniform_reference":{"marginal_probability":UNIFORM_P,"mean_top6_hits_expected":6*6/45},
        "metrics":metrics,
        "interpretation":{"predictive_advantage_established":False,
          "rule":"No candidate is promoted from this experiment; require adjusted significance, CI direction, stability across chronological subperiods, and independent prospective confirmation."}
    }


def run_v93(csv_path: str|Path, output: str|Path, bootstrap_reps: int=BOOTSTRAP_REPS) -> dict[str,Any]:
    draws,numbers,fingerprint=load_draws(csv_path)
    report=analyze_v93(draws,numbers,bootstrap_reps)
    report["data"]={"sha256":fingerprint,"draw_count":len(draws)}
    dest=Path(output)
    if dest.exists(): raise FileExistsError(f"拒绝覆盖已有研究报告: {dest}")
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv",default="data/lotto645.csv"); ap.add_argument("--output",default="artifacts/v93_multiscale_report.json")
    ap.add_argument("--bootstrap-reps",type=int,default=BOOTSTRAP_REPS)
    args=ap.parse_args(); result=run_v93(args.csv,args.output,args.bootstrap_reps)
    print(json.dumps(result,ensure_ascii=False,indent=2))
