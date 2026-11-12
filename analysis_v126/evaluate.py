"""v1.26独立评价进程：加载审计标签后只输出分数与汇总诊断。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from analysis_v126.data import FRACTIONS, ROLE_SEEDS, S_REF, ids_sha256, load_target_features, read_json, sha256
from g2f_design.metrics import ranking_metrics


def unique_strings(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "US" or len(array) != len(set(array.tolist())):
        raise ValueError(f"{name}身份不是唯一一维字符串")
    return array.astype(str)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_auc(standardized_regrets, budgets, query_count):
    """输入已标准化，积分时不再除s_ref。"""
    y = np.asarray(standardized_regrets, dtype=np.float64)
    counts = np.asarray(budgets, dtype=np.float64)
    require(y.shape == (4,) and counts.shape == (4,), "四预算主曲线不完整")
    require(query_count > 0 and np.isfinite(y).all() and np.isfinite(counts).all(), "AUC输入非法")
    require(np.all(counts == np.floor(counts)) and np.all(counts > 0) and np.all(counts <= query_count), "AUC预算非法")
    x = counts / query_count
    require(np.all(np.diff(x) > 0), "实际预算横轴不严格递增")
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2) / (x[-1] - x[0]))


def upper_cvar(values, alpha=.8):
    ordered = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    require(len(ordered) > 0 and np.isfinite(ordered).all(), "CVaR损失非法")
    mass = (1 - alpha) * len(ordered)
    weights = np.clip(mass - np.arange(len(ordered)), 0, 1)
    return float(np.dot(ordered, weights) / mass)


def load_audit(audit_path, target, role):
    expected = {"combo_ids", "env", "hybrid_ids", "y_raw", "target_indices", "sref", "center"}
    with np.load(audit_path, allow_pickle=False) as archive:
        require(set(archive.files) == expected, "审计标签字段不符合合同")
        audit = {key: archive[key] for key in archive.files}
    ids = unique_strings(audit["combo_ids"], "审计")
    role_ids = unique_strings(role["audit_ids"], "角色审计")
    require(set(ids) == set(role_ids), "审计标签与角色C不一致")
    n = len(ids)
    require(all(audit[key].shape == (n,) for key in ("env", "hybrid_ids", "y_raw", "target_indices")), "审计数组未对齐")
    require(audit["target_indices"].dtype.kind in "iu", "审计行号不是整数")
    indices = audit["target_indices"]
    require(np.all(indices >= 0) and np.all(indices < len(target["combo_ids"])), "审计行号越界")
    require(np.array_equal(target["combo_ids"][indices], ids), "审计行号与组合身份不一致")
    require(np.array_equal(target["env"][indices], audit["env"]) and np.array_equal(target["hybrid_ids"][indices], audit["hybrid_ids"]), "审计环境或品种身份不一致")
    require(audit["y_raw"].dtype == np.float64 and np.isfinite(audit["y_raw"]).all(), "审计原始标签必须为有限float64")
    require(audit["center"].shape == () and audit["sref"].shape == (), "审计尺度不是标量")
    require(math.isfinite(float(audit["center"])) and np.isclose(float(audit["sref"]), S_REF, atol=1e-12, rtol=0), "审计尺度错误")
    order = {identity: i for i, identity in enumerate(ids)}
    aligned = np.asarray([order[identity] for identity in role_ids], dtype=np.int64)
    return {key: value[aligned] if value.shape == (n,) else value for key, value in audit.items()}


def evaluate_role(data_dir, prediction_dir, seed, *, target=None):
    """评价一个角色的四预算预测；返回可复用于外层年份的汇总，不返回标签。"""
    data_dir, prediction_dir = Path(data_dir), Path(prediction_dir)
    role_path = data_dir / "roles" / f"s{seed}" / "role.json"
    audit_path = data_dir / "evaluation" / f"s{seed}" / "audit_labels.npz"
    target = load_target_features(data_dir / "target_features.npz") if target is None else target
    year = int(target["outer_year"])
    role = read_json(role_path)
    require(role["seed"] == seed and role["year"] == year, "角色年度或种子错误")
    query_ids = unique_strings(role["query_ids"], "查询")
    audit_ids = unique_strings(role["audit_ids"], "审计")
    initial_ids = unique_strings(role["initial_ids"], "初始支持")
    query_set, audit_set = set(query_ids), set(audit_ids)
    require(not query_set & audit_set and query_set | audit_set == set(target["combo_ids"]), "C/U不构成完整不相交候选池")
    require(set(initial_ids).issubset(query_set), "S0不属于U")
    audit = load_audit(audit_path, target, role)
    center, sref = float(audit["center"]), float(audit["sref"])
    target_ids_hash = ids_sha256(target["combo_ids"].tolist())
    role_hash = sha256(role_path)
    source_by_fraction = {item["fraction"]: item for item in role["M_sources"]}
    require(len(source_by_fraction) == 4 and set(source_by_fraction) == set(FRACTIONS), "原M四预算凭据不完整")
    budget_results = []
    for fraction in FRACTIONS:
        support_path = role_path.parent / f"support_b{fraction}.json"
        support = read_json(support_path)
        selected = unique_strings(support["ids"], "支持")
        budget = int(support["budget"])
        expected_budget = int(round(float(fraction) * 100)) * len(query_ids) // 100
        require(support["budget"] == budget == len(selected) == role["budgets"][fraction] == expected_budget, "实际预算不符")
        require(support["year"] == year and support["seed"] == seed and support["fraction"] == fraction, "支持身份错误")
        require(support["role_sha256"] == role_hash and support["ids_sha256"] == ids_sha256(selected.tolist()), "支持或角色散列错误")
        require(support["target_ids_sha256"] == target_ids_hash, "支持的目标候选池错误")
        require(set(selected).issubset(query_set) and not set(selected) & audit_set and set(initial_ids).issubset(selected), "支持越权或遗漏S0")
        require(len(support["labels"]) == budget and np.isfinite(np.asarray(support["labels"], dtype=float)).all(), "支持标签数量或数值错误")
        require(support["label_unit"] == "historical_standardized" and np.isclose(support["sref"], sref, atol=1e-12, rtol=0) and np.isclose(support["center"], center, atol=1e-12, rtol=0), "支持/审计的中心或尺度不一致")
        source = source_by_fraction[fraction]
        require(source["budget"] == budget and source["selected_ids_sha256"] == support["ids_sha256"], "支持不对应原M选样")
        pred_path = prediction_dir / f"b{fraction}.npz"
        receipt_path = prediction_dir / f"b{fraction}.receipt.json"
        receipt = read_json(receipt_path)
        require(receipt["year"] == year and receipt["seed"] == seed and receipt["fraction"] == fraction and receipt["budget"] == budget, "预测实际预算凭据错误")
        require(receipt["role_sha256"] == role_hash and receipt["support_sha256"] == sha256(support_path) and receipt["prediction_sha256"] == sha256(pred_path), "预测与支持文件关联错误")
        require(receipt["label_unit"] == "historical_standardized", "预测单位不是历史标准化")
        with np.load(pred_path, allow_pickle=False) as archive:
            require(set(archive.files) == {"combo_ids", "prediction"}, "预测只能含身份和成员均值")
            pred_ids = unique_strings(archive["combo_ids"], "预测")
            prediction = np.asarray(archive["prediction"], dtype=np.float64)
        require(set(pred_ids) == audit_set and prediction.shape == (len(pred_ids),) and np.isfinite(prediction).all(), "预测未完整对应C或数值错误")
        lookup = {identity: i for i, identity in enumerate(pred_ids)}
        aligned = np.asarray([lookup[identity] for identity in audit["combo_ids"]], dtype=np.int64)
        raw_prediction = center + sref * prediction[aligned]
        require(np.isfinite(raw_prediction).all(), "恢复原始单位后预测非有限")
        env_results = []
        for env in np.unique(audit["env"]):
            mask = audit["env"] == env
            y, p = audit["y_raw"][mask], raw_prediction[mask]
            metrics = ranking_metrics(y, p, audit["hybrid_ids"][mask].tolist())
            metrics.update({"environment": str(target["env_names"][int(env)]), "count": int(mask.sum()),
                            "standardized_regret": metrics["regret"] / sref,
                            "centered_rmse": float(np.sqrt(np.mean(((y - y.mean()) - (p - p.mean())) ** 2)))})
            env_results.append(metrics)
        regrets = [m["standardized_regret"] for m in env_results]
        finite_correlations = [m["spearman"] for m in env_results if m["spearman"] is not None]
        budget_results.append({
            "fraction": fraction, "budget": budget, "query_count": len(query_ids), "actual_fraction": budget / len(query_ids),
            "macro_standardized_regret": float(np.mean(regrets)),
            "macro_retention": float(np.mean([m["retention"] for m in env_results])),
            "macro_rmse": float(np.mean([m["rmse"] for m in env_results])),
            "macro_centered_rmse": float(np.mean([m["centered_rmse"] for m in env_results])),
            "global_rmse": float(np.sqrt(np.mean((audit["y_raw"] - raw_prediction) ** 2))),
            "macro_spearman": float(np.mean(finite_correlations)) if finite_correlations else None,
            "undefined_spearman_environments": len(env_results) - len(finite_correlations),
            "environment_cvar_0.8": upper_cvar(regrets), "environment_metrics": env_results,
            "prediction_sha256": receipt["prediction_sha256"], "prediction_receipt_sha256": sha256(receipt_path),
            "support_sha256": receipt["support_sha256"], "support_ids_sha256": support["ids_sha256"],
        })
    score = normalized_auc([row["macro_standardized_regret"] for row in budget_results], [row["budget"] for row in budget_results], len(query_ids))
    return {"year": year, "seed": seed, "score": score, "score_name": "actual_budget_normalized_standardized_regret_AUC",
            "macro_retention": float(np.mean([row["macro_retention"] for row in budget_results])),
            "budgets": budget_results, "role_sha256": role_hash, "audit_file_sha256": sha256(audit_path),
            "audit_ids_sha256": ids_sha256(audit_ids.tolist()), "sref": sref,
            "evaluation_only_labels_returned": False}


def evaluate_year(data_dir, predictions):
    data_dir, predictions = Path(data_dir), Path(predictions)
    target = load_target_features(data_dir / "target_features.npz")
    roles = [evaluate_role(data_dir, predictions / f"s{seed}", seed, target=target) for seed in ROLE_SEEDS]
    return {"state": "EXECUTION_VALID", "year": int(target["outer_year"]),
            "score": float(np.mean([row["score"] for row in roles])),
            "score_name": "five_role_mean_actual_budget_normalized_standardized_regret_AUC",
            "macro_retention": float(np.mean([row["macro_retention"] for row in roles])),
            "role_seeds": list(ROLE_SEEDS), "roles": roles,
            "target_file_sha256": sha256(data_dir / "target_features.npz"),
            "evaluation_only_labels_returned": False,
            "evidence_level": "开发评分" if int(target["outer_year"]) == 2019 else "探索性外层评价"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_year(args.data_dir, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("state", "year", "score", "macro_retention")}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
