# -*- coding: utf-8 -*-
"""Model registry and version metadata for Lotto 6/45 V3.6."""
from __future__ import annotations

ENGINE_VERSION = "3.6.0-phase1"
MODEL_DEFINITIONS = [
    {
        "name": "近期频率模型",
        "key": "recent_frequency",
        "version": "1.1",
        "family": "频率统计",
        "description": "基于最近指定训练窗口的号码频率，并进行弱先验平滑。",
        "status": "实验",
    },
    {
        "name": "长期频率模型",
        "key": "long_frequency",
        "version": "1.1",
        "family": "频率统计",
        "description": "使用测试期之前的全部可用历史开奖记录，而非仅使用近期训练窗口。",
        "status": "实验",
    },
    {
        "name": "指数时间衰减模型",
        "key": "exponential_decay",
        "version": "1.1",
        "family": "时间衰减",
        "description": "根据历史记录距测试期的期数差进行指数衰减加权。",
        "status": "实验",
    },
    {
        "name": "贝叶斯平滑模型",
        "key": "bayesian_smoothing",
        "version": "1.1",
        "family": "贝叶斯平滑",
        "description": "以理论频率为先验中心，对号码频率进行平滑。",
        "status": "实验",
    },
    {
        "name": "综合评分模型",
        "key": "ensemble_score",
        "version": "1.1",
        "family": "集成统计",
        "description": "融合短期、中期及全部历史频率统计。",
        "status": "实验",
    },
    {
        "name": "历史结构约束模型",
        "key": "historical_structure",
        "version": "1.1",
        "family": "结构筛选",
        "description": "先均匀抽样，再按训练历史中的宽松和值及分区条件筛选。",
        "status": "实验",
    },
    {
        "name": "均匀随机基准",
        "key": "uniform_random",
        "version": "1.0",
        "family": "对照基准",
        "description": "从45个号码中等概率、不放回抽取6个号码，永久保留为对照。",
        "status": "基准",
    },
]
MODEL_NAMES = [item["name"] for item in MODEL_DEFINITIONS]


def get_model_registry() -> list[dict[str, str]]:
    """Return a defensive copy suitable for display/export."""
    return [dict(item) for item in MODEL_DEFINITIONS]


def get_model_version(model_name: str) -> str:
    for item in MODEL_DEFINITIONS:
        if item["name"] == model_name:
            return item["version"]
    raise KeyError(f"未注册的模型：{model_name}")
