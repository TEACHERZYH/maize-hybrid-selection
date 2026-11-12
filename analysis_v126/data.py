"""v1.26 训练入口：只读取历史标签及当前允许支持标签。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

S_REF = 2.78561424436278
ROLE_SEEDS = (11, 23, 37, 53, 71)
FRACTIONS = ("0.1", "0.2", "0.4", "0.6")
FEATURE_KEYS = {"x", "env", "combo_ids", "hybrid_ids", "genetic_pc", "env_names", "outer_year", "x_center", "x_scale"}
HISTORY_KEYS = FEATURE_KEYS | {"y", "year", "center", "sref"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_npz(path, keys):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != keys:
            raise ValueError("数据字段不符合标签隔离合同")
        arrays = {key: saved[key] for key in saved.files}
    n = len(arrays["combo_ids"])
    if len(set(arrays["combo_ids"].tolist())) != n:
        raise ValueError("组合身份重复")
    if arrays["x"].ndim != 2 or not 1 <= arrays["x"].shape[1] <= 296:
        raise ValueError("输入维数不符合合同")
    for key in ("x", "env", "hybrid_ids", "genetic_pc"):
        if len(arrays[key]) != n:
            raise ValueError("输入行数不一致")
    for key in ("x", "genetic_pc"):
        expected_dtype = np.float32 if key == "x" else np.float64
        if arrays[key].dtype != expected_dtype or not np.isfinite(arrays[key]).all():
            raise ValueError("网络输入必须为有限float32，桥点PCA必须保留float64")
    if any(arrays[key].shape != (arrays["x"].shape[1],) or not np.isfinite(arrays[key]).all() for key in ("x_center", "x_scale")) or np.any(arrays["x_scale"] <= 0):
        raise ValueError("历史特征标准化参数非法")
    if arrays["env"].dtype.kind not in "iu" or np.any(arrays["env"] < 0) or np.any(arrays["env"] >= len(arrays["env_names"])):
        raise ValueError("环境索引非法")
    return arrays


def load_history(path, outer_year):
    data = _load_npz(path, HISTORY_KEYS)
    if int(data["outer_year"]) != outer_year or np.any(data["year"] >= outer_year):
        raise ValueError("历史包含当年或未来标签")
    if len(data["y"]) != len(data["x"]) or len(data["year"]) != len(data["x"]):
        raise ValueError("历史标签/年份未对齐")
    if data["y"].dtype != np.float32 or not np.isfinite(data["y"]).all():
        raise ValueError("历史标签必须为有限float32")
    if not np.isclose(float(data["sref"]), S_REF, atol=1e-12, rtol=0) or not np.isfinite(data["center"]):
        raise ValueError("历史产量尺度非法")
    return data


def load_target_features(path):
    return _load_npz(path, FEATURE_KEYS)


def load_support(path, target, role_path):
    """只接受一个预算标签文件；角色文件只含身份，不含隐藏产量。"""
    data, role = read_json(path), read_json(role_path)
    ids = data["ids"]
    if len(ids) != len(set(ids)) or len(ids) != data["budget"] or len(ids) != len(data["labels"]):
        raise ValueError("支持数量或身份重复")
    if data["year"] != int(target["outer_year"]) or data["seed"] != role["seed"]:
        raise ValueError("支持角色错误")
    if data["role_sha256"] != sha256(role_path) or data["ids_sha256"] != ids_sha256(ids):
        raise ValueError("支持身份散列不一致")
    if data["target_ids_sha256"] != ids_sha256(target["combo_ids"].tolist()):
        raise ValueError("目标身份散列不一致")
    if not set(ids).issubset(role["query_ids"]) or set(ids) & set(role["audit_ids"]):
        raise ValueError("支持越权读取审计标签")
    if data["budget"] and not set(role["initial_ids"]).issubset(ids):
        raise ValueError("支持集缺少既有S0")
    if data["budget"] != role["budgets"][data["fraction"]]:
        raise ValueError("支持预算不符")
    if data["label_unit"] != "historical_standardized" or not np.isclose(data["sref"], S_REF, atol=1e-12, rtol=0):
        raise ValueError("支持尺度不符合合同")
    lookup = {key: i for i, key in enumerate(target["combo_ids"].tolist())}
    indices = np.asarray([lookup[key] for key in ids], dtype=np.int64)
    y = np.asarray(data["labels"], dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("支持标签非有限")
    return {"indices": indices, "y": y, "ids": np.asarray(ids), "receipt": data}


def load_training_inputs(history_path, target_path, support_path, role_path, outer_year):
    history = load_history(history_path, outer_year)
    target = load_target_features(target_path)
    if int(target["outer_year"]) != outer_year:
        raise ValueError("目标年份与外层任务不一致")
    support = load_support(support_path, target, role_path)
    if any(not np.array_equal(history[key], target[key]) for key in ("env_names", "x_center", "x_scale")) or history["x"].shape[1] != target["x"].shape[1]:
        raise ValueError("历史和目标编码不一致")
    if not np.isclose(float(history["center"]), support["receipt"]["center"], atol=1e-12, rtol=0):
        raise ValueError("支持中心与历史中心不一致")
    return {"history": history, "target": target, "support": support}


def eligible_episode_years(history):
    """训练A可用全部历史；训练B只返回≥100组合且年度≥2环境的任务。"""
    eligible = {}
    for year in sorted(set(history["year"].tolist())):
        mask = history["year"] == year
        envs = [int(env) for env in np.unique(history["env"][mask])
                if int(np.sum(mask & (history["env"] == env))) >= 100]
        if len(envs) >= 2:
            eligible[int(year)] = envs
    return eligible
