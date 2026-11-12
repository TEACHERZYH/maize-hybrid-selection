"""v1.26 外层预测：绑定固定停止点及来源，仅读允许的支持标签。"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from analysis_v126.data import (
    FRACTIONS, ROLE_SEEDS, ids_sha256, load_history, load_support,
    load_target_features, read_json, sha256,
)
from analysis_v126.model import ResponseTabM, adapt_predict
from analysis_v126.train import configure_runtime, scientific_hashes, validate_prediction_metadata, write_json

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "C1": ("ordinary", "intercept", FRACTIONS),
    "C2": ("ordinary", "response", FRACTIONS),
    "C3": ("centered", "response", FRACTIONS),
    "N1": ("N1", "response", FRACTIONS),
    "N1_INT": ("N1", "intercept", FRACTIONS),
    "C2_ZERO": ("ordinary", "zero", ("0.0",)),
    "N1_ZERO": ("N1", "zero", ("0.0",)),
    "C2_ALL": ("ordinary", "response", ("1.0",)),
    "N1_ALL": ("N1", "response", ("1.0",)),
}


def _public_relative(relative):
    """白名单不包含 evaluation；连隐藏标签的散列也不在预测进程计算。"""
    relative = str(relative).replace("\\", "/")
    allowed = {"history.npz", "target_features.npz", "export_receipt.json"}
    for seed in ROLE_SEEDS:
        allowed.add(f"roles/s{seed}/role.json")
        allowed.update(f"roles/s{seed}/support_b{f}.json" for f in ("0.0", *FRACTIONS, "1.0"))
    if relative not in allowed:
        raise ValueError("预测进程禁止读取未列入白名单的文件")
    return relative


class SourceContract:
    """2024 必须使用已修复来源；其它年度保留原隔离导出。"""

    def __init__(self, data_dir, year, registry_path=None):
        self.data_dir, self.year = Path(data_dir).resolve(), year
        self.registry, self.registry_sha256 = None, None
        self.matrix_path = ROOT / "design_v126/planned_runs.csv"
        if year == 2024:
            registry_path = Path(registry_path or ROOT / "reports/v126/R0_2024_repaired/execution_sources_R2.json")
            registry = read_json(registry_path)
            if (registry.get("state") != "PASS_EXECUTION_SOURCES" or registry.get("version") != "v1.26-R2"
                    or registry.get("year") != year):
                raise ValueError("2024 缺少有效的 R2 来源清单")
            if self.data_dir != Path(registry["required_2024_data_dir"]).resolve():
                raise ValueError("2024 只能使用 R2 清单指定的新数据目录")
            if self.data_dir != Path(registry["new_export_directory"]).resolve():
                raise ValueError("R2 来源清单内部目录不一致")
            self.registry, self.registry_sha256 = registry, sha256(registry_path)
            self.matrix_path = Path(registry["revised_matrix"]["path"])
            if sha256(self.matrix_path) != registry["revised_matrix"]["sha256"]:
                raise ValueError("R2 实验矩阵散列不一致")
        self.checked = {}
        for relative in ("history.npz", "target_features.npz", "export_receipt.json"):
            self.verify(relative)
        receipt = read_json(self.data_dir / "export_receipt.json")
        if receipt.get("state") != "PASS" or receipt.get("year") != year:
            raise ValueError("数据导出未通过或年度不一致")
        if (receipt["history_file_sha256"] != self.checked["history.npz"]
                or receipt["target_file_sha256"] != self.checked["target_features.npz"]):
            raise ValueError("导出回执与输入数组散列不一致")
        if self.registry and (receipt["prepared_sha256"] != self.registry["prepared"]["sha256"]
                              or receipt["environment_availability_sha256"] != self.registry["availability"]["sha256"]):
            raise ValueError("2024 导出仍引用旧 prepared 或可用性清单")

    def verify(self, relative):
        relative = _public_relative(relative)
        digest = sha256(self.data_dir / relative)
        if self.registry and self.registry["new_export_files"].get(relative) != digest:
            raise ValueError(f"2024 新来源散列不一致：{relative}")
        self.checked[relative] = digest
        return self.data_dir / relative

    def validate_m_source(self, role, fraction, support):
        if fraction not in FRACTIONS:
            return
        matches = [m for m in role["M_sources"] if m["fraction"] == fraction]
        if len(matches) != 1:
            raise ValueError("角色缺少唯一的 M 选样来源")
        source = matches[0]
        if source["budget"] != len(support["ids"]) or source["selected_ids_sha256"] != ids_sha256(support["ids"].tolist()):
            raise ValueError("支持身份与 M 选样清单不一致")
        if self.registry:
            run_id = f"C0_y2024_r{role['seed']}_tNA_b{fraction}"
            matched = [m for m in self.registry["new20M_C0"] if m["run_id"] == run_id]
            if len(matched) != 1:
                raise ValueError("R2 清单缺少该角色预算的新 M 来源")
            expected = matched[0]["new_source"]
            actual_path = Path(source["source"])
            if not actual_path.is_absolute():
                actual_path = ROOT / actual_path
            if actual_path.resolve() != Path(expected["path"]).resolve() or source["sha256"] != expected["sha256"]:
                raise ValueError("2024 支持仍继承旧 M 来源")


def validate_role(role, target, year, role_seed):
    if role["year"] != year or role["seed"] != role_seed:
        raise ValueError("角色年份或种子不一致")
    query, audit = role["query_ids"], role["audit_ids"]
    if (len(query) != len(set(query)) or len(audit) != len(set(audit)) or set(query) & set(audit)
            or set(query) | set(audit) != set(target["combo_ids"].tolist())
            or not set(role["initial_ids"]).issubset(query)):
        raise ValueError("U/C 角色划分不构成目标全集的无交叠分区")


def validate_support_request(support, role, fraction):
    if support["receipt"]["fraction"] != fraction:
        raise ValueError("支持文件与请求预算不一致")
    if fraction == "0.0" and len(support["ids"]):
        raise ValueError("零标签参照不能提供 S0")
    if fraction == "1.0" and set(support["ids"].tolist()) != set(role["query_ids"]):
        raise ValueError("全查询参照必须恰好使用全部 U")


def validate_stopping_contract(development, completed, year, seed, branch):
    if (development.get("state") != "COMPLETED" or development.get("outer_year") != 2019
            or development.get("seed") != 1009 or development.get("stage") != branch):
        raise ValueError("停止点必须来自完成的 2019 对应分支开发轨迹")
    epoch = development["best_epoch"]
    observed = development["completed_epoch"]
    if (type(epoch) is not int or not 0 <= epoch <= 30 or not epoch <= observed <= 30
            or not (observed == 30 or (observed >= 5 and development["stopping"].get("stop") is True
                                      and development["stopping"].get("no_improvement", 0) >= 5))):
        raise ValueError("2019 开发轨迹尚未达到固定停止条件")
    if branch == "N1" and epoch == 0:
        raise ValueError("N1 停止点为 0，不能启动外层扩展")
    if (completed.get("state") != "COMPLETED" or completed.get("outer_year") != year
            or completed.get("seed") != seed or completed.get("stage") != branch
            or completed.get("completed_epoch") != epoch or completed.get("best_epoch") != epoch):
        raise ValueError("外层训练没有在该分支固定 E_m 完成")
    return epoch


def load_bound_model(checkpoint, development_path, stage_result_path, source, year, seed, branch, projection):
    checkpoint, projection = Path(checkpoint), Path(projection)
    epoch = validate_stopping_contract(read_json(development_path), read_json(stage_result_path), year, seed, branch)
    checkpoint_hash = sha256(checkpoint)
    state_path = checkpoint.with_suffix(".state.json")
    if read_json(state_path).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("检查点尚未收尾或散列不一致")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if any(payload.get(key) != value for key, value in {
        "format": "v1.26", "stage": branch, "outer_year": year, "seed": seed,
        "epoch": epoch, "steps_per_epoch": 100,
    }.items()):
        raise ValueError("检查点年份、网络种子、分支或 E_m 不一致")
    # 在同一冻结执行快照内核全部科学项，包括损失/训练代码和计划，不能只核网络结构。
    expected = scientific_hashes(source.data_dir, projection, branch)
    expected["development_stage_result.json"] = sha256(development_path)
    for role_seed in ROLE_SEEDS:
        for filename in ("role.json", *(f"support_b{f}.json" for f in FRACTIONS)):
            relative = f"roles/s{role_seed}/{filename}"
            if expected[relative] != sha256(source.verify(relative)):
                raise ValueError("核验过程中公开科学输入发生改变")
    if any(payload["scientific_files_sha256"].get(key) != value for key, value in expected.items()):
        raise ValueError("预测输入或开发停止点与训练检查点不一致")
    fixed = np.load(projection, allow_pickle=False)
    if fixed.shape != (128, 8) or fixed.dtype != np.float32 or not np.isfinite(fixed).all():
        raise ValueError("固定投影格式不符合合同")
    model = ResponseTabM(payload["input_dim"], seed)
    model.load_state_dict(payload["model"], strict=True)
    if (not np.array_equal(model.projection.detach().numpy(), fixed)
            or model.projection_sha256 != payload["projection_sha256"]):
        raise ValueError("检查点内固定投影被更改")
    binding = {"year": year, "training_seed": seed, "checkpoint_family": branch, "E_m": epoch,
               "checkpoint_sha256": checkpoint_hash, "checkpoint_state_sha256": sha256(state_path),
               "development_result_sha256": sha256(development_path), "stage_result_sha256": sha256(stage_result_path),
               "projection_sha256": model.projection_sha256, "execution_sources_sha256": source.registry_sha256,
               "history_sha256": expected["history.npz"], "target_features_sha256": expected["target_features.npz"],
               "prediction_code_sha256": sha256(Path(__file__)), "member_reduction": "mean_16"}
    return model, binding


def planned_identities(matrix_path, methods, year, training_seed, branch):
    with Path(matrix_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for method in methods:
        family, _, fractions = METHODS[method]
        if family != branch:
            raise ValueError("请求方法与检查点分支不一致")
        for role_seed in ROLE_SEEDS:
            for fraction in fractions:
                matrix_fraction = {"0.0": "0", "1.0": "1"}.get(fraction, fraction)
                matches = [row for row in rows if row["method"] == method and row["year"] == str(year)
                           and row["role_seed"] == str(role_seed) and row["training_seed"] == str(training_seed)
                           and row["budget_fraction"] == matrix_fraction and row["checkpoint_family"] == branch
                           and row["stage"] == "R2"]
                if len(matches) != 1:
                    raise ValueError("预测身份未唯一列入已审核实验矩阵")
                selected[(method, role_seed, fraction)] = matches[0]["run_id"]
    if len(set(selected.values())) != len(selected):
        raise ValueError("实验矩阵存在重复运行身份")
    return selected


def _save_prediction(path, audit_ids, prediction, expected, diagnostics):
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path, diagnostic_path = path.with_suffix(".receipt.json"), path.with_suffix(".diagnostics.json")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (any(receipt.get(k) != v for k, v in expected.items()) or not path.exists()
                or sha256(path) != receipt["prediction_sha256"] or not diagnostic_path.exists()
                or sha256(diagnostic_path) != receipt["diagnostics_sha256"]):
            raise ValueError("已有预测与当前运行绑定不一致，禁止覆盖历史证据")
        return receipt
    for unfinished in (path, diagnostic_path):
        if unfinished.exists():
            unfinished.replace(unfinished.with_name(f"{unfinished.name}.incomplete_{time.time_ns()}"))
    values = prediction.detach().cpu().numpy()
    if values.shape != (len(audit_ids),) or not np.isfinite(values).all():
        raise ValueError("预测必须为有限的一维成员均值")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, combo_ids=np.asarray(audit_ids), prediction=values)
    temporary.replace(path)
    write_json(diagnostic_path, diagnostics)
    receipt = {**expected, "prediction_sha256": sha256(path), "diagnostics_sha256": sha256(diagnostic_path)}
    write_json(receipt_path, receipt)
    return receipt


def predict_methods(model, history, target, source, output, methods, binding, identities, device="cpu", chunk_size=512):
    """每个检查点只提取一次全部 V；每个角色/预算独立求解，禁止跨网络种子合并。"""
    output = Path(output)
    validate_prediction_metadata(history, target)
    if model.input_dim != target["x"].shape[1] or int(target["outer_year"]) != binding["year"]:
        raise ValueError("目标输入维数或年份与检查点不一致")
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "execution_binding.json"
    if binding_path.exists() and read_json(binding_path) != binding:
        raise ValueError("输出目录已属于不同检查点、种子或来源")
    write_json(binding_path, binding)
    model.to(device).eval()
    lookup = {key: i for i, key in enumerate(target["combo_ids"].tolist())}
    records = []
    with torch.no_grad():
        base, phi = model.forward_chunked(torch.as_tensor(target["x"], device=device), chunk_size, checkpoint_chunks=False)
        env = torch.as_tensor(target["env"], dtype=torch.long, device=device)
        for role_seed in ROLE_SEEDS:
            role_relative = f"roles/s{role_seed}/role.json"
            role_path = source.verify(role_relative)
            role = read_json(role_path)
            validate_role(role, target, binding["year"], role_seed)
            audit_indices = torch.as_tensor([lookup[key] for key in role["audit_ids"]], dtype=torch.long, device=device)
            for method in methods:
                family, mode, fractions = METHODS[method]
                if family != binding["checkpoint_family"]:
                    raise ValueError("方法与训练分支不一致")
                for fraction in fractions:
                    started = time.perf_counter()
                    relative = f"roles/s{role_seed}/support_b{fraction}.json"
                    support_path = source.verify(relative)
                    support = load_support(support_path, target, role_path)
                    validate_prediction_metadata(history, target, support)
                    validate_support_request(support, role, fraction)
                    source.validate_m_source(role, fraction, support)
                    prediction, diagnostics = adapt_predict(
                        base, phi, env, torch.as_tensor(support["indices"], dtype=torch.long, device=device),
                        torch.as_tensor(support["y"], dtype=torch.float32, device=device),
                        mode=mode, lambda_a=1.0, lambda_w=10.0,
                    )
                    expected = {**binding, "run_id": identities[(method, role_seed, fraction)],
                                "method": method, "mode": mode, "seed": role_seed, "fraction": fraction,
                                "budget": len(support["ids"]), "role_sha256": source.checked[role_relative],
                                "support_sha256": source.checked[relative], "label_unit": "historical_standardized"}
                    path = output / method / f"s{role_seed}" / f"b{fraction}.npz"
                    receipt = _save_prediction(path, role["audit_ids"], prediction.mean(1)[audit_indices], expected, diagnostics)
                    records.append({"run_id": receipt["run_id"], "prediction_path": str(path.resolve()),
                                    "receipt_sha256": sha256(path.with_suffix(".receipt.json")),
                                    "seconds": time.perf_counter() - started})
    manifest = {"state": "PREDICTIONS_READY", **binding, "methods": list(methods), "count": len(records),
                "records": records, "model_readable_files_sha256": source.checked,
                "evaluation": "预测已保存；本进程未读取隐藏标签或评价科学主张"}
    # 不同方法子集使用不同清单；同一预测身份不能覆盖为另一方法或来源。
    manifest_name = "predictions_" + "_".join(methods) + ".json"
    write_json(output / manifest_name, manifest)
    return manifest


def run(args):
    if args.year not in range(2020, 2025) or args.training_seed not in (1009, 2027, 3037):
        raise ValueError("外层年份或网络种子不在已审核清单内")
    if args.chunk_size < 1:
        raise ValueError("分块大小必须为正")
    methods = tuple(args.methods or [method for method, spec in METHODS.items() if spec[0] == args.branch])
    if not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError("方法必须来自已审核清单且不能重复")
    configure_runtime(args.training_seed, args.device, args.threads)
    source = SourceContract(args.data_dir, args.year, args.r2_sources)
    identities = planned_identities(source.matrix_path, methods, args.year, args.training_seed, args.branch)
    if source.registry and not set(methods).issubset(source.registry["all_2024_methods"]):
        raise ValueError("2024 方法未列入新来源清单")
    stage_result_path = args.stage_result or Path(args.checkpoint).parent / "stage_result.json"
    model, binding = load_bound_model(args.checkpoint, args.development_result, stage_result_path,
                                     source, args.year, args.training_seed, args.branch, args.projection)
    binding["matrix_sha256"] = sha256(source.matrix_path)
    history = load_history(source.data_dir / "history.npz", args.year)
    target = load_target_features(source.data_dir / "target_features.npz")
    return predict_methods(model, history, target, source, args.output, methods, binding, identities,
                           args.device, args.chunk_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--branch", choices=("ordinary", "centered", "N1"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--development-result", required=True)
    parser.add_argument("--stage-result")
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS))
    parser.add_argument("--r2-sources")
    parser.add_argument("--projection", default=str(ROOT / "design_v126/projection.npy"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    result = run(parser.parse_args())
    print(__import__("json").dumps({key: result[key] for key in ("state", "year", "training_seed", "checkpoint_family", "E_m", "count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
