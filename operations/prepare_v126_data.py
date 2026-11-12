"""v1.26 可信导出进程：核对旧身份并把训练与审计标签写入不同文件。"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from analysis_v126.data import (FRACTIONS, ROLE_SEEDS, S_REF, eligible_episode_years,
                               ids_sha256, load_history, load_support, load_target_features, read_json, sha256)
from g2f_design.dataio import combination_values, ledger_rows, read_plots
from g2f_design.geometry import bridge_set


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"不覆盖不同证据：{path}")
    path.write_text(text, encoding="utf-8")


def combo_id(row):
    return json.dumps([row["env"], row["hybrid"]], ensure_ascii=False, separators=(",", ":"))


def parse_source_date(value):
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError("来源日期格式不能可靠解析")


def paths_for_year(year, args):
    contract = ROOT / "data_contract" / ("S1_2024" if year == 2024 else "S1_H")
    if year < 2020:
        prepared = ROOT / f"reports/S3/checkpoints/prepared_{year}.pkl"
        receipt = ROOT / f"reports/S3/remote_preparation/prepared_{year}.json"
    elif year == 2024:
        prepared = ROOT / "runs/S4_P1_2024_assumed/prepared_2024.pkl"
        receipt = prepared.with_suffix(".json")
    else:
        prepared = ROOT / f"reports/S4/P1_completion_evidence/runs/S4_B6/prepared_{year}.pkl"
        receipt = prepared.with_suffix(".json")
    if args.prepared_root:
        prepared = Path(args.prepared_root) / f"prepared_{year}.pkl"
        receipt = prepared.with_suffix(".json")
    if args.contract:
        contract = Path(args.contract)
    return contract, prepared, receipt


def role_path_for_year(year, args, contract):
    return (Path(args.role_root) if args.role_root else contract) / f"P1_{year}_roles.json"


def m_path(year, seed, fraction, args):
    if year < 2020:
        folder = Path(args.development_m_root)
    elif year == 2024 and args.m2024_root:
        folder = Path(args.m2024_root)
    else:
        folder = ROOT / "reports/S4" / ("P1_2024_synced_results" if year == 2024 else "P1_synced_results")
    return folder / f"P1_{year}_M_s{seed}_b{fraction}.json"


def validate_m_group(roles, role_hash, results, year, seed):
    pool = [json.dumps(key, ensure_ascii=False, separators=(",", ":")) for key in roles["pool"]]
    if len(pool) != len(set(pool)):
        raise ValueError("角色池组合重复")
    role = roles["seeds"][str(seed)]
    audit = [pool[i] for i in role["audit_indices"]]
    audit_indices = set(role["audit_indices"])
    query = [key for i, key in enumerate(pool) if i not in audit_indices]
    if len(query) != role["query_count"] or len(audit) != len(set(audit)):
        raise ValueError("C/U角色数量错误")
    counts = Counter(key[0] for key in roles["pool"])
    audit_counts = Counter(json.loads(key)[0] for key in audit)
    if any(n < 100 or audit_counts[env] != (3 * n + 9) // 10 for env, n in counts.items()):
        raise ValueError("环境资格或审计比例不符")
    initial, previous = None, set()
    receipts = []
    for fraction in FRACTIONS:
        row, source = results[fraction]
        expected = {"state": "PASS", "year": year, "seed": seed, "method": "M", "fraction": fraction}
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("M记录身份或执行状态不符")
        selected, s0 = row["selected_ids"], row["initial_ids"]
        budget = int(round(float(fraction) * 100)) * len(query) // 100
        if row["budget"] != budget or budget != role["budgets"][fraction] or len(selected) != budget or len(set(selected)) != budget:
            raise ValueError("M预算或唯一性错误")
        if not set(selected).issubset(query) or set(selected) & set(audit):
            raise ValueError("M支持访问审计角色")
        if len(s0) != role["S0_size"] or len(s0) != len(set(s0)) or not set(s0).issubset(selected):
            raise ValueError("M初始集合错误")
        if initial is None:
            initial = s0
        if initial != s0 or not previous.issubset(selected):
            raise ValueError("M预算不嵌套或S0改变")
        if row["audit_ids"] != audit or row["role_file_sha256"] != role_hash:
            raise ValueError("M审计身份/角色散列错误")
        if not np.isclose(row["s_ref"], S_REF, atol=1e-12, rtol=0):
            raise ValueError("M历史尺度错误")
        previous = set(selected)
        receipts.append({"fraction": fraction, "budget": budget, "source": str(source),
                         "sha256": sha256(source), "selected_ids_sha256": ids_sha256(selected)})
    return {"year": year, "seed": seed, "query_ids": query, "audit_ids": audit,
            "initial_ids": initial, "budgets": {"0.0": 0, "1.0": len(query), **{b: role["budgets"][b] for b in FRACTIONS}},
            "source_role_sha256": role_hash, "M_sources": receipts}


def audit_year(year, args):
    contract, prepared, receipt = paths_for_year(year, args)
    role_file = role_path_for_year(year, args, contract)
    missing = [str(p) for p in (role_file, prepared, receipt) if not p.exists()]
    missing += [str(m_path(year, s, b, args)) for s in ROLE_SEEDS for b in FRACTIONS if not m_path(year, s, b, args).exists()]
    if missing:
        return {"year": year, "state": "WAITING_RESOURCE", "missing": missing}, None
    roles = read_json(role_file)
    verified = {}
    for seed in ROLE_SEEDS:
        results = {b: (read_json(m_path(year, seed, b, args)), m_path(year, seed, b, args)) for b in FRACTIONS}
        verified[seed] = validate_m_group(roles, sha256(role_file), results, year, seed)
    summary = {"year": year, "state": "PASS", "M_count": 20, "role_sha256": sha256(role_file),
               "prepared_path": str(prepared), "prepared_receipt": str(receipt),
               "groups": [{"seed": s, "query_count": len(v["query_ids"]), "audit_count": len(v["audit_ids"]), "M_sources": v["M_sources"]} for s, v in verified.items()]}
    return summary, verified


def feature_arrays(records, fmap, env_names, outer_year):
    env_lookup = {name: i for i, name in enumerate(env_names)}
    return {"x": np.column_stack(fmap.components(records)),
            "env": np.asarray([env_lookup[r["env"]] for r in records], dtype=np.int64),
            "combo_ids": np.asarray([combo_id(r) for r in records]),
            "hybrid_ids": np.asarray([r["hybrid"] for r in records]),
            "genetic_pc": np.asarray([fmap.genetic[r["hybrid"]] for r in records], dtype=np.float64),
            "env_names": np.asarray(env_names), "outer_year": np.asarray(outer_year, dtype=np.int64)}


def save_npz(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise FileExistsError(f"不覆盖不同数组：{path}")
        return
    np.savez(path, **arrays)


def export_year(year, args, verified):
    contract, prepared, receipt_path = paths_for_year(year, args)
    r2_binding = None
    if year == 2024:
        if not args.r2_receipt or not args.m2024_root:
            raise ValueError("2024须显式指定经R2修复的输入回执与新M目录，不能直接继承旧2024")
        r2_binding = read_json(args.r2_receipt)
        expected = {"repaired_prepared_sha256": sha256(prepared),
                    "repaired_availability_sha256": sha256(contract / "environment_availability.json"),
                    "role_sha256": sha256(role_path_for_year(year, args, contract))}
        if r2_binding.get("state") != "PASS" or r2_binding.get("version") != "v1.26-R2" or any(r2_binding.get(k) != v for k, v in expected.items()):
            raise ValueError("R2输入回执与实际修复文件不一致")
        if expected["repaired_prepared_sha256"] == "547bbfccfc1eea9f3ada287588f66b64d07f33cb1577329a318aa44a228fba86":
            raise ValueError("拒绝将旧2024输入冒充修复版")
        old_m_root = (ROOT / "reports/S4/P1_2024_synced_results").resolve()
        if Path(args.m2024_root).resolve() == old_m_root:
            raise ValueError("2024修复必须使用新的M结果目录")
        if set(verified) != set(ROLE_SEEDS):
            raise ValueError("2024实际导出须等待全20份修复M/C0验收")
        for seed in ROLE_SEEDS:
            for fraction in FRACTIONS:
                m = read_json(m_path(year, seed, fraction, args))
                if (m.get("model_artifact_sha256") != r2_binding["repaired_prepared_sha256"]
                        or m.get("version") != "v1.26-R2" or not m.get("execution_valid")
                        or m.get("source_r2_receipt_sha256") != sha256(args.r2_receipt)):
                    raise ValueError("2024支持未通过修复版M/C0验收")
    receipt = read_json(receipt_path)
    if receipt["state"] != "PASS" or sha256(prepared) != receipt["artifact_sha256"]:
        raise ValueError("旧prepared完整性核对失败")
    # 唯一允许读取旧pickle的可信导出进程；训练器只读取新npz和单预算JSON。
    with prepared.open("rb") as handle:
        payload = pickle.load(handle)
    roles = read_json(role_path_for_year(year, args, contract))
    if payload["year"] != year or payload["roles"] != roles:
        raise ValueError("prepared角色与独立角色文件不一致")
    history_contract = contract if (contract / "combination_cost_ledger.csv").exists() else ROOT / "data_contract/S1_H"
    history_rows = [r for r in ledger_rows(history_contract, maximum_year=year - 1)]
    if [combo_id(r) for r in history_rows] != payload["history_ids"]:
        raise ValueError("历史组合顺序不一致")
    hist_values = combination_values(history_rows, read_plots(args.raw, history_rows))
    center = float(hist_values.mean())
    sref = float(np.std(hist_values[np.asarray([r["year"] <= 2017 for r in history_rows])], ddof=1))
    if not np.isclose(center, payload["center"], atol=1e-12, rtol=0) or not np.isclose(sref, S_REF, atol=1e-12, rtol=0):
        raise ValueError("原始历史标签不复现已验收尺度")
    fmap = payload["schema"].feature_map
    env_names = sorted({r["env"] for r in history_rows + payload["current"]})
    whitelist = read_json(contract / "feature_whitelist.json")
    availability = {r["env"]: r for r in read_json(contract / "environment_availability.json")}
    if fmap.environment_pca.source_width != len(whitelist["geographic_columns"]) + len(whitelist["soil_columns"]) + 24:
        raise ValueError("历史缓存环境列数与原白名单不符")
    climate_exceptions = []
    approved_2024 = {"MOH1_2024": [2017, 2019, 2023], "NEH2_2024": [2020, 2021, 2022], "OHH1_2024": [2020, 2022, 2023]}
    for env in env_names:
        rule = availability[env]
        planted = parse_source_date(rule.get("first_planting_date"))
        soil = parse_source_date(rule.get("soil_report_date"))
        truly_available = bool(planted and soil and soil < planted)
        if truly_available != rule["soil_available_before_first_planting"]:
            raise ValueError(f"土壤报告实际日期与播前可用标志不符：{env}；不能继承时间越界输入")
        years = [availability[source]["year"] for source in rule["historical_climate_source_envs"]]
        if any(y >= rule["year"] for y in years):
            raise ValueError("环境气候摘要引用当年或未来数据")
        if any(y < rule["year"] - 3 for y in years):
            if not (r2_binding is not None and year == 2024 and years == approved_2024.get(env)):
                raise ValueError("环境气候摘要超出前三个日历年；仅允许R2回执绑定的2024三处精确例外")
            climate_exceptions.append({"env": env, "source_years": years, "lags_years": [rule["year"]-y for y in years]})
    history = feature_arrays(history_rows, fmap, env_names, year)
    history.update(y=((hist_values - center) / sref).astype(np.float32),
                   year=np.asarray([r["year"] for r in history_rows], dtype=np.int64),
                   center=np.asarray(center), sref=np.asarray(sref))
    target = feature_arrays(payload["current"], fmap, env_names, year)
    x_center = history["x"].mean(axis=0)
    x_scale = history["x"].std(axis=0, ddof=0)
    constant = np.ptp(history["x"], axis=0) == 0
    if np.any((x_scale == 0) & ~constant):
        raise FloatingPointError("真实变化列标准差下溢")
    x_scale[constant] = 1.0
    for data in (history, target):
        data["x"] = ((data["x"] - x_center) / x_scale).astype(np.float32)
        data["x"][:, constant] = 0.0
        data["x_center"], data["x_scale"] = x_center.copy(), x_scale.copy()
    if target["combo_ids"].tolist() != [json.dumps(key, ensure_ascii=False, separators=(",", ":")) for key in roles["pool"]]:
        raise ValueError("目标数组没有按角色池排序")
    out = Path(args.output) / str(year)
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "history.npz", history)
    save_npz(out / "target_features.npz", target)
    load_history(out / "history.npz", year)
    checked_target = load_target_features(out / "target_features.npz")
    lookup = {key: i for i, key in enumerate(target["combo_ids"].tolist())}
    labels = np.asarray(payload["current_values"], dtype=np.float64)
    for seed, role in verified.items():
        role_dir = out / "roles" / f"s{seed}"
        write_json(role_dir / "role.json", role)
        query_indices = [lookup[key] for key in role["query_ids"]]
        bridge = bridge_set(target["env"][query_indices], target["hybrid_ids"][query_indices], role["query_ids"],
                            np.asarray([fmap.genetic[h] for h in target["hybrid_ids"][query_indices]]))
        if [role["query_ids"][i] for i in bridge] != role["initial_ids"]:
            raise ValueError("既有S0与原桥点定义不一致")
        for fraction in ("0.0",) + FRACTIONS + ("1.0",):
            ids = ([] if fraction == "0.0" else role["query_ids"] if fraction == "1.0"
                   else read_json(m_path(year, seed, fraction, args))["selected_ids"])
            support = {"year": year, "seed": seed, "fraction": fraction, "budget": len(ids), "ids": ids,
                       "labels": ((labels[[lookup[key] for key in ids]] - center) / sref).tolist(),
                       "label_unit": "historical_standardized", "center": center, "sref": sref,
                       "ids_sha256": ids_sha256(ids), "target_ids_sha256": ids_sha256(target["combo_ids"].tolist()),
                       "role_sha256": sha256(role_dir / "role.json")}
            path = role_dir / f"support_b{fraction}.json"
            write_json(path, support)
            load_support(path, checked_target, role_dir / "role.json")
    # 审计导出与支持名单生成独立，任何训练入口均不调用本部分。
    for seed in ROLE_SEEDS:
        evaluation = out / "evaluation" / f"s{seed}"
        evaluation.mkdir(parents=True, exist_ok=True)
        ai = np.asarray(roles["seeds"][str(seed)]["audit_indices"], dtype=np.int64)
        save_npz(evaluation / "audit_labels.npz", {"combo_ids": target["combo_ids"][ai], "env": target["env"][ai],
                 "hybrid_ids": target["hybrid_ids"][ai], "y_raw": labels[ai], "target_indices": ai, "sref": np.asarray(sref), "center": np.asarray(center)})
    result = {"state": "PASS", "year": year, "history_rows": len(history_rows), "target_rows": len(labels),
              "feature_width": int(history["x"].shape[1]), "center": center, "sref": sref,
              "historical_episode_years": eligible_episode_years(load_history(out / "history.npz", year)),
              "prepared_sha256": receipt["artifact_sha256"], "trait_source_sha256": sha256(Path(args.raw) / "1_Training_Trait_Data_2014_2023.csv"),
              "feature_whitelist_sha256": sha256(contract / "feature_whitelist.json"),
              "environment_availability_sha256": sha256(contract / "environment_availability.json"),
              "history_ids_sha256": ids_sha256(history["combo_ids"].tolist()),
              "history_file_sha256": sha256(out / "history.npz"), "target_file_sha256": sha256(out / "target_features.npz"),
              "x_standardization": "296列按历史组合均值和总体标准差拟合；历史常数列固定为零；参数随文件保存，训练器不得二次拟合。",
              "PCA_inheritance": "继承已验收prepared中仅按历史唯一实体拟合的分量，不重新拟合；genetic/env/外积不含截距或随机效应编码。",
              "label_isolation": "训练入口只加载history、target_features、一个预算support及不含标签role；审计产量仅在evaluation。"}
    write_json(out / "export_receipt.json", result)
    write_json(out / "source_validation.json", {"state": "PASS", "year": year,
               "historical_ledger_sha256": sha256(history_contract / "combination_cost_ledger.csv"),
               "climate_window_check": ("全部来源严格早于目标年；无例外环境均位于前三个日历年；2024例外仅允许R2绑定的精确三环境年份。" if year == 2024
                                         else "全部来源严格早于目标年；无例外环境均位于前三个日历年；2024例外遵照v1.26-R1原样继承。"),
               "climate_exceptions": climate_exceptions,
               "feature_whitelist_sha256": sha256(contract / "feature_whitelist.json"),
               "environment_availability_sha256": sha256(contract / "environment_availability.json")})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2019, 2025)))
    parser.add_argument("--raw", default="G:/datasets/G2F_2024_competition/raw/Training_data")
    parser.add_argument("--output", default=str(ROOT / "reports/v126/R0_data_exports"))
    parser.add_argument("--development-m-root", default=str(ROOT / "reports/v126/R0_data_development_M"))
    parser.add_argument("--2024-m-root", dest="m2024_root")
    parser.add_argument("--r2-receipt")
    parser.add_argument("--prepared-root")
    parser.add_argument("--contract")
    parser.add_argument("--role-root")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--receipt", default=str(ROOT / "reports/v126/R0_data_audit.json"))
    args = parser.parse_args()
    results = []
    for year in args.years:
        if args.features_only:
            export = export_year(year, args, {})
            results.append({"year": year, "state": "FEATURES_EXPORTED_SUPPORT_PENDING", "export": export})
            print(json.dumps({"year": year, "state": "FEATURES_EXPORTED_SUPPORT_PENDING"}), flush=True)
            continue
        audit, verified = audit_year(year, args)
        if verified is not None and not args.audit_only:
            audit["export"] = export_year(year, args, verified)
        results.append(audit)
        print(json.dumps({"year": year, "state": audit["state"], "M_count": audit.get("M_count", 0)}, ensure_ascii=False), flush=True)
    write_json(args.receipt, {"state": "PASS" if all(r["state"] == "PASS" for r in results) else "WAITING_RESOURCE", "years": results})


if __name__ == "__main__":
    main()
