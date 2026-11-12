"""v1.26 固定训练轨迹；历史标签训练，2019 停止点评价由独立进程执行。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from g2f_design.geometry import bridge_set
from g2f_design.metrics import ranking_metrics

from .data import FRACTIONS, ROLE_SEEDS, eligible_episode_years, load_history, load_support, load_target_features, read_json, sha256
from .model import ResponseTabM, adapt_predict

STEPS_PER_EPOCH = 100
PRETRAIN_EPOCHS = 20
TRAINING_SEEDS = (1009, 2027, 3037)


def task_rng(outer_year: int, seed: int, step: int, phase: str):
    """任务流不含方法名；恢复后同一步生成完全相同的历史任务。"""
    key = f"v1.26|{outer_year}|{seed}|{step}|{phase}"
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], "little"))


def env_groups(history, indices=None):
    indices = np.arange(len(history["x"])) if indices is None else np.asarray(indices)
    return {int(e): indices[history["env"][indices] == e] for e in sorted(np.unique(history["env"][indices]))}


def sample_pretrain(history, outer_year: int, seed: int, step: int):
    rng = task_rng(outer_year, seed, step, "A")
    groups = env_groups(history)
    chosen = rng.choice(sorted(groups), size=min(16, len(groups)), replace=False)
    return np.concatenate([rng.choice(groups[int(e)], size=min(64, len(groups[int(e)])), replace=False) for e in chosen])


@dataclass
class Episode:
    year: int
    fraction: float
    pool: np.ndarray
    query: np.ndarray
    target: np.ndarray
    support: np.ndarray
    initial: np.ndarray
    loss_target: np.ndarray


def sample_episode(history, outer_year: int, seed: int, step: int) -> Episode:
    rng = task_rng(outer_year, seed, step, "B")
    eligible = eligible_episode_years(history)
    if not eligible:
        raise ValueError("历史无符合条件的阶段 B 任务年份")
    year = int(rng.choice(sorted(eligible)))
    pool = np.flatnonzero((history["year"] == year) & np.isin(history["env"], eligible[year]))
    pool = pool[np.argsort(history["combo_ids"][pool], kind="stable")]
    query, target, loss_target = [], [], []
    for e in sorted(eligible[year]):
        indices = np.flatnonzero(history["env"][pool] == e)
        shuffled = rng.permutation(indices)
        n_target = (3 * len(indices) + 9) // 10
        target.append(shuffled[:n_target])
        query.append(shuffled[n_target:])
        loss_target.append(rng.choice(shuffled[:n_target], size=min(64, n_target), replace=False))
    query, target = np.concatenate(query), np.concatenate(target)
    if len(query) == 0 or len(target) == 0:
        raise ValueError("阶段 B 查询池或目标池为空")
    initial_query = np.asarray(bridge_set(
        history["env"][pool[query]], history["hybrid_ids"][pool[query]],
        history["combo_ids"][pool[query]], history["genetic_pc"][pool[query]],
    ), dtype=np.int64)
    initial = query[initial_query]
    percent = int(rng.choice([10, 20, 40, 60]))
    fraction = percent / 100
    budget = percent * len(query) // 100
    if budget < len(initial):
        raise ValueError("既有桥点数超过整数预算；禁止自动扩大预算")
    remaining = np.setdiff1d(query, initial)
    support = np.concatenate([initial, rng.choice(remaining, budget - len(initial), replace=False)])
    return Episode(year, fraction, pool, query, target, support, initial, np.concatenate(loss_target))


def equal_environment_loss(prediction: Tensor, labels: Tensor, environments: Tensor, centered=False):
    """成员先在环境内均方，再对环境及成员等权；C3 再加批次中心化误差。"""
    error = prediction - labels[:, None]
    losses = []
    for e in torch.unique(environments):
        values = error[environments == e]
        loss = values.square().mean(0)
        if centered:
            loss = loss + (values - values.mean(0)).square().mean(0)
        losses.append(loss)
    if not losses:
        raise ValueError("损失目标为空")
    return torch.stack(losses).mean()


def _as_tensor(array, device, dtype=None):
    return torch.as_tensor(array, device=device, dtype=dtype)


def _json_value(value):
    if isinstance(value, Tensor):
        return _json_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def append_log(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False) + "\n")


def _training_diagnostics(labels, before, after, environments, identifiers):
    result = {"training_regret_before": [], "training_regret_after": [], "within_environment_mse": []}
    y, p0, p1 = labels.detach().cpu().numpy(), before.detach().mean(1).cpu().numpy(), after.detach().mean(1).cpu().numpy()
    env = np.asarray(environments)
    for e in np.unique(env):
        mask = env == e
        result["training_regret_before"].append(ranking_metrics(y[mask], p0[mask], identifiers[mask])["regret"])
        result["training_regret_after"].append(ranking_metrics(y[mask], p1[mask], identifiers[mask])["regret"])
        error = p1[mask] - y[mask]
        result["within_environment_mse"].append(float(np.square(error - error.mean()).mean()))
    return {key: float(np.mean(values)) for key, values in result.items()}


def optimization_step(model, optimizer, history, stage, outer_year, seed, step, device="cpu", chunk_size=512, diagnose=False):
    """执行一个固定训练步；测试可直接用合成历史调用，不更改 CLI 科学设置。"""
    started = time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    diagnostics: dict[str, Any] = {}
    if stage == "pretrain":
        indices = sample_pretrain(history, outer_year, seed, step)
        x = _as_tensor(history["x"][indices], device, torch.float32)
        labels = _as_tensor(history["y"][indices], device, torch.float32)
        env = _as_tensor(history["env"][indices], device, torch.long)
        base, _ = model.forward_chunked(x, chunk_size)
        loss = equal_environment_loss(base, labels, env)
        if diagnose:
            diagnostics.update(_training_diagnostics(labels, base, base, history["env"][indices], history["combo_ids"][indices]))
        diagnostics["rows"] = len(indices)
    else:
        episode = sample_episode(history, outer_year, seed, step)
        indices = episode.pool
        x = _as_tensor(history["x"][indices], device, torch.float32)
        env = _as_tensor(history["env"][indices], device, torch.long)
        base, phi = model.forward_chunked(x, chunk_size)
        support = _as_tensor(episode.support, device, torch.long)
        support_labels = _as_tensor(history["y"][indices[episode.support]], device, torch.float32)
        target = _as_tensor(episode.loss_target, device, torch.long)
        target_labels = _as_tensor(history["y"][indices[episode.loss_target]], device, torch.float32)
        if stage == "N1":
            adapted, diagnostics = adapt_predict(base, phi, env, support, support_labels)
            prediction = adapted
        elif stage in {"ordinary", "centered"}:
            prediction = base
            if diagnose:
                with torch.no_grad():
                    adapted, diagnostics = adapt_predict(base.detach(), phi.detach(), env, support, support_labels)
        else:
            raise ValueError("未知训练阶段")
        loss = equal_environment_loss(prediction[target], target_labels, env[target], centered=stage == "centered")
        if diagnose:
            diagnostics.update(_training_diagnostics(target_labels, base[target], adapted[target], history["env"][indices[episode.loss_target]], history["combo_ids"][indices[episode.loss_target]]))
        diagnostics.update({"episode_year": episode.year, "fraction": episode.fraction, "support_count": len(episode.support), "pool_rows": len(indices), "target_rows": len(target)})
    if not torch.isfinite(loss):
        raise FloatingPointError("训练损失非有限；保留当前日志后停止本分支")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach().cpu()), "gradient_norm_before_clip": float(grad.detach().cpu()), "seconds": time.perf_counter() - started, **diagnostics}


def make_optimizer(model, stage):
    return torch.optim.AdamW(model.parameters(), lr=1e-3 if stage == "pretrain" else 3e-4,
                            weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)


def eligibility_manifest(history):
    eligible = eligible_episode_years(history)
    entries = []
    for year in sorted(set(history["year"].tolist())):
        for env, indices in env_groups(history, np.flatnonzero(history["year"] == year)).items():
            included = year in eligible and env in eligible[year]
            reason = "纳入" if included else "环境不足100组合" if len(indices) < 100 else "年度不足两个合格环境"
            entries.append({"year": int(year), "env": env, "rows": len(indices), "stage_B_eligible": included, "reason": reason})
    return entries


def save_checkpoint(path, model, optimizer, *, stage, epoch, seed, outer_year, hashes, stopping):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有epoch证据：{path}")
    payload = {
        "format": "v1.26", "stage": stage, "epoch": epoch, "steps_per_epoch": STEPS_PER_EPOCH,
        "seed": seed, "outer_year": outer_year, "input_dim": model.input_dim,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "projection_sha256": model.projection_sha256, "scientific_files_sha256": hashes,
        "operations_files_sha256": {}, "stopping": stopping,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, *, stage, seed, outer_year, hashes=None, inherit_A=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format"] != "v1.26" or payload["seed"] != seed or payload["outer_year"] != outer_year:
        raise ValueError("检查点版本、网络种子或拟合年份不一致")
    if inherit_A:
        if payload["stage"] != "pretrain" or payload["epoch"] != PRETRAIN_EPOCHS:
            raise ValueError("阶段 B 必须继承完整20epoch的阶段 A 权重和优化器")
    elif payload["stage"] != stage:
        raise ValueError("不能把不同 B 方法的检查点作为恢复点")
    if hashes is not None:
        old_hashes = payload["scientific_files_sha256"]
        if inherit_A:
            if any(key not in hashes or hashes[key] != value for key, value in old_hashes.items()):
                raise ValueError("阶段 A 共享科学输入已发生变化")
        elif hashes != old_hashes:
            raise ValueError("恢复时科学输入哈希发生变化")
    if payload["projection_sha256"] != model.projection_sha256:
        raise ValueError("固定投影与检查点不一致")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if inherit_A:
        for group in optimizer.param_groups:
            group["lr"] = 3e-4
    random.setstate(payload["rng"]["python"])
    np.random.set_state(payload["rng"]["numpy"])
    torch.set_rng_state(payload["rng"]["torch"])
    if payload["rng"]["cuda"] is not None and next(model.parameters()).is_cuda:
        torch.cuda.set_rng_state_all(payload["rng"]["cuda"])
    state_path = Path(path).with_suffix(".state.json")
    if state_path.exists():
        state = read_json(state_path)
        if state["checkpoint_sha256"] != sha256(path):
            raise ValueError("检查点收尾状态与权重哈希不符")
        payload["stopping"] = state["stopping"]
    payload["epoch_finalized"] = state_path.exists()
    return payload


def update_stopping(stopping, score, epoch):
    if not np.isfinite(score):
        raise ValueError("开发评价得分非有限")
    result = dict(stopping)
    if result.get("best_score") is None or score < result["best_score"] - 1e-8:
        result.update(best_score=score, best_epoch=epoch, no_improvement=0)
    else:
        result["no_improvement"] = result.get("no_improvement", 0) + 1
    result["last_score"] = score
    result["stop"] = epoch >= 5 and result["no_improvement"] >= 5
    return result


def completion_state(stage, outer_year, completed, stopping):
    if stage == "pretrain":
        return "COMPLETED" if completed == PRETRAIN_EPOCHS else "CHECKPOINT_READY"
    if outer_year == 2019:
        return "COMPLETED" if stopping["stop"] or completed == 30 else "CHECKPOINT_READY"
    return "COMPLETED"


def validate_prediction_metadata(history, target, support=None):
    if int(history["outer_year"]) != int(target["outer_year"]):
        raise ValueError("历史和目标拟合年份不一致")
    if history["x"].shape[1] != target["x"].shape[1] or any(
        not np.array_equal(history[key], target[key]) for key in ("env_names", "x_center", "x_scale")
    ):
        raise ValueError("历史与目标的环境编码或特征标准化不一致")
    if support is not None and (not np.isclose(float(history["center"]), support["receipt"]["center"], atol=1e-12, rtol=0)
                                or not np.isclose(float(history["sref"]), support["receipt"]["sref"], atol=1e-12, rtol=0)):
        raise ValueError("支持标签标准化与历史尺度不一致")


def write_predictions(model, data_dir, directory, device, chunk_size, history=None):
    """仅读目标特征及各预算支持标签；输出隐藏身份对应预测供独立评价器使用。"""
    data_dir, directory = Path(data_dir), Path(directory)
    target = load_target_features(data_dir / "target_features.npz")
    if history is None:
        history = load_history(data_dir / "history.npz", int(target["outer_year"]))
    validate_prediction_metadata(history, target)
    model.eval()
    lookup = {str(key): index for index, key in enumerate(target["combo_ids"].tolist())}
    model_digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        model_digest.update(key.encode())
        model_digest.update(value.detach().cpu().numpy().tobytes())
    model_hash = model_digest.hexdigest()
    with torch.no_grad():
        base, phi = model.forward_chunked(_as_tensor(target["x"], device, torch.float32), chunk_size, checkpoint_chunks=False)
        env = _as_tensor(target["env"], device, torch.long)
        for role_seed in ROLE_SEEDS:
            role_path = data_dir / "roles" / f"s{role_seed}" / "role.json"
            role = read_json(role_path)
            audit_ids = np.asarray(role["audit_ids"])
            audit_indices = np.asarray([lookup[str(key)] for key in audit_ids])
            output = directory / f"s{role_seed}"
            output.mkdir(parents=True, exist_ok=True)
            for fraction in FRACTIONS:
                support_path = role_path.parent / f"support_b{fraction}.json"
                support = load_support(support_path, target, role_path)
                validate_prediction_metadata(history, target, support)
                path = output / f"b{fraction}.npz"
                receipt_path = path.with_suffix(".receipt.json")
                expected = {
                    "year": int(target["outer_year"]), "seed": role_seed, "fraction": fraction,
                    "budget": support["receipt"]["budget"], "role_sha256": sha256(role_path),
                    "support_sha256": sha256(support_path), "checkpoint_model_sha256": model_hash,
                    "label_unit": "historical_standardized",
                }
                if path.exists():
                    if receipt_path.exists():
                        receipt = read_json(receipt_path)
                        if any(receipt.get(key) != value for key, value in expected.items()) or receipt["prediction_sha256"] != sha256(path):
                            raise ValueError("已存预测不属于同一模型或标签预算，禁止覆盖")
                        continue
                    # 只保留尚未形成回执的中断文件，不能把它当有效证据复用。
                    path.replace(path.with_suffix(f".incomplete_{time.time_ns()}.npz"))
                prediction, diagnostic = adapt_predict(base, phi, env, _as_tensor(support["indices"], device, torch.long), _as_tensor(support["y"], device, torch.float32))
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, combo_ids=audit_ids, prediction=prediction.mean(1)[audit_indices].cpu().numpy())
                temporary.replace(path)
                write_json(receipt_path, {**expected, "prediction_sha256": sha256(path)})
                write_json(path.with_suffix(".diagnostics.json"), diagnostic)
    return directory


def evaluate_development(model, data_dir, output, epoch, device, chunk_size, history=None):
    predictions = Path(output) / "development" / f"epoch_{epoch:03d}"
    write_predictions(model, data_dir, predictions, device, chunk_size, history)
    summary = predictions / "evaluation.json"
    # 命令仅传路径，隐藏标签从未载入训练进程。
    process = subprocess.run(
        [sys.executable, str(ROOT / "analysis_v126" / "evaluate.py"),
         "--data-dir", str(data_dir), "--predictions", str(predictions), "--output", str(summary)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if process.returncode != 0:
        (predictions / "evaluator_failure.log").write_text(process.stderr[-8000:], encoding="utf-8")
        raise RuntimeError("独立开发评价失败，详见 evaluator_failure.log；不能据此修改模型")
    return read_json(summary)


def scientific_hashes(data_dir, projection, stage):
    sources = [
        "36_稀疏观测响应学习实验计划_v1.26.md", "design_v126/model_design.md",
        "design_v126/active_execution_reference.json", "design_v126/revisions/v1.26-R1_data_time_window.md",
        "analysis_v126/model.py", "analysis_v126/train.py", "analysis_v126/data.py",
        "analysis_v126/vendor/tabm.py", "src/g2f_design/geometry.py", "src/g2f_design/metrics.py",
    ]
    hashes = {name: sha256(ROOT / name) for name in sources}
    hashes["projection.npy"] = sha256(projection)
    hashes["history.npz"] = sha256(Path(data_dir) / "history.npz")
    if stage != "pretrain":
        hashes["analysis_v126/evaluate.py"] = sha256(ROOT / "analysis_v126/evaluate.py")
        hashes["target_features.npz"] = sha256(Path(data_dir) / "target_features.npz")
        for seed in ROLE_SEEDS:
            for filename in ["role.json"] + [f"support_b{fraction}.json" for fraction in FRACTIONS]:
                key = f"roles/s{seed}/{filename}"
                hashes[key] = sha256(Path(data_dir) / key)
    return hashes


def configure_runtime(seed, device, threads):
    if threads < 1:
        raise ValueError("threads 必须为正")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求CUDA但环境无可用GPU；不自动改变已选资源")


def run(args):
    data_dir, output = Path(args.data_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with np.load(data_dir / "history.npz", allow_pickle=False) as metadata:
        outer_year = int(metadata["outer_year"])
    if outer_year not in (2019, 2020, 2021, 2022, 2023, 2024):
        raise ValueError("拟合年份不在v1.26实验清单内")
    if args.seed not in TRAINING_SEEDS or (outer_year == 2019 and args.seed != 1009):
        raise ValueError("网络种子不在当前年份的固定清单内")
    if args.stage == "pretrain":
        if not 1 <= args.epochs <= PRETRAIN_EPOCHS or args.pretrain_checkpoint:
            raise ValueError("A阶段仅允许逐步完成固定20epoch，不继承另一个A")
    elif not 0 <= args.epochs <= 30 or (outer_year == 2019 and args.epochs < 5):
        raise ValueError("B阶段epoch超出固定设计；开发至少5最多30")
    if args.stage == "N1" and outer_year > 2019 and args.epochs == 0:
        raise ValueError("N1开发E=0触发停止，不得启动外层N1")
    if args.stage != "pretrain" and not (args.pretrain_checkpoint or args.resume_checkpoint):
        raise ValueError("阶段B必须明确指定完整A检查点")
    if args.pretrain_checkpoint and args.resume_checkpoint:
        raise ValueError("初始继承和恢复不能同时指定")
    if outer_year > 2019 and args.stage != "pretrain":
        if not args.development_result:
            raise ValueError("外层 B 必须提供2019开发停止点文件")
        development_result = read_json(args.development_result)
        if (development_result["stage"] != args.stage or development_result["outer_year"] != 2019
                or development_result["seed"] != 1009 or development_result["state"] != "COMPLETED"
                or development_result["best_epoch"] != args.epochs):
            raise ValueError("外层 B 训练长度或方法与2019开发停止点不一致")
    configure_runtime(args.seed, args.device, args.threads)
    history = load_history(data_dir / "history.npz", outer_year)
    hashes = scientific_hashes(data_dir, args.projection, args.stage)
    if args.stage != "pretrain":
        target = load_target_features(data_dir / "target_features.npz")
        validate_prediction_metadata(history, target)
        for role_seed in ROLE_SEEDS:
            role_path = data_dir / "roles" / f"s{role_seed}" / "role.json"
            for fraction in FRACTIONS:
                support = load_support(role_path.parent / f"support_b{fraction}.json", target, role_path)
                validate_prediction_metadata(history, target, support)
    if args.development_result:
        hashes["development_stage_result.json"] = sha256(args.development_result)
    model = ResponseTabM(history["x"].shape[1], args.seed)
    projection = np.load(args.projection, allow_pickle=False)
    if projection.dtype != np.float32 or projection.shape != (128, 8) or not np.isfinite(projection).all():
        raise ValueError("固定投影文件必须为float32[128,8]")
    model.projection.copy_(torch.from_numpy(projection))
    model.to(args.device)
    optimizer = make_optimizer(model, args.stage)
    stopping = {"best_score": None, "best_epoch": None, "no_improvement": 0, "stop": False}
    start_epoch, pending_finalization = 0, False
    if args.resume_checkpoint:
        payload = load_checkpoint(args.resume_checkpoint, model, optimizer, stage=args.stage, seed=args.seed, outer_year=outer_year, hashes=hashes)
        start_epoch, stopping = payload["epoch"], payload["stopping"]
        pending_finalization = not payload["epoch_finalized"]
    elif args.pretrain_checkpoint:
        load_checkpoint(args.pretrain_checkpoint, model, optimizer, stage=args.stage, seed=args.seed, outer_year=outer_year, hashes=hashes, inherit_A=True)
    elif list(output.glob("checkpoint_*.pt")):
        raise FileExistsError("输出已有检查点，请使用明确恢复入口")
    if args.epochs < start_epoch:
        raise ValueError("请求完成epoch小于已有检查点")
    development = outer_year == 2019 and args.stage != "pretrain"
    receipt_path = output / "input_receipt.json"
    if receipt_path.exists():
        receipt_path = output / f"resume_receipt_{time.time_ns()}.json"
    write_json(receipt_path, {
        "stage": args.stage, "outer_year": outer_year, "seed": args.seed, "epochs_requested": args.epochs,
        "scientific_files_sha256": hashes, "operations_files_sha256": {},
        "projection_sha256": model.projection_sha256, "device": args.device, "threads": args.threads,
        "chunk_size": args.chunk_size, "python": sys.version.split()[0], "torch": torch.__version__,
        "network_dtype": "float32", "solve_dtype": "float64", "AMP": False, "TF32": False,
        "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
        "parameters": sum(p.numel() for p in model.parameters()),
        "data_directory": str(data_dir.resolve()), "history_rows": len(history["x"]),
    })
    write_json(output / "episode_eligibility.json", eligibility_manifest(history))

    def finalize(epoch, checkpoint_path, state):
        if development:
            evaluation = evaluate_development(model, data_dir, output, epoch, args.device, args.chunk_size, history)
            state = update_stopping(state, evaluation["score"], epoch)
            append_log(output / "epoch_metrics.jsonl", {"epoch": epoch, "development": evaluation, "stopping": state})
        write_json(Path(checkpoint_path).with_suffix(".state.json"), {"checkpoint_sha256": sha256(checkpoint_path), "stopping": state})
        return state

    if not args.resume_checkpoint:
        initial = output / "checkpoint_000.pt"
        save_checkpoint(initial, model, optimizer, stage=args.stage, epoch=0, seed=args.seed, outer_year=outer_year, hashes=hashes, stopping=stopping)
        stopping = finalize(0, initial, stopping)
    elif pending_finalization:
        stopping = finalize(start_epoch, args.resume_checkpoint, stopping)
    completed = start_epoch
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if development and stopping["stop"]:
            break
        started = time.perf_counter()
        losses, gradients, step_seconds = [], [], []
        for offset in range(STEPS_PER_EPOCH):
            step = (epoch - 1) * STEPS_PER_EPOCH + offset
            diagnose = (offset + 1) % 10 == 0
            metrics = optimization_step(model, optimizer, history, args.stage, outer_year, args.seed, step, args.device, args.chunk_size, diagnose)
            losses.append(metrics["loss"])
            gradients.append(metrics["gradient_norm_before_clip"])
            step_seconds.append(metrics["seconds"])
            if diagnose:
                append_log(output / "step_metrics.jsonl", {"epoch": epoch, "step": step + 1, "learning_rate": optimizer.param_groups[0]["lr"], **metrics})
        epoch_summary = {"epoch": epoch, "training_loss": float(np.mean(losses)), "gradient_norm_mean": float(np.mean(gradients)), "gradient_norm_max": float(np.max(gradients)), "step_seconds_mean": float(np.mean(step_seconds)), "training_seconds": time.perf_counter() - started}
        append_log(output / "epoch_metrics.jsonl", epoch_summary)
        checkpoint_path = output / f"checkpoint_{epoch:03d}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, stage=args.stage, epoch=epoch, seed=args.seed, outer_year=outer_year, hashes=hashes, stopping=stopping)
        stopping = finalize(epoch, checkpoint_path, stopping)
        completed = epoch
        print(json.dumps({"stage": args.stage, **epoch_summary, "best_epoch": stopping["best_epoch"]}, ensure_ascii=False), flush=True)
    status = completion_state(args.stage, outer_year, completed, stopping)
    final = {"state": status, "stage": args.stage, "outer_year": outer_year, "seed": args.seed, "completed_epoch": completed,
             "best_epoch": stopping["best_epoch"] if development else completed, "stopping": stopping,
             "scientific_result": "本文件仅记录执行；不表示正向主张成立"}
    if development and args.stage == "N1" and status == "COMPLETED" and stopping["best_epoch"] == 0:
        final["downstream_state"] = "WAITING_AUTHORIZATION"
        final["reason"] = "N1开发停止点为0；按计划不启动后续N1外层训练"
    write_json(output / "stage_result.json", final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("pretrain", "ordinary", "centered", "N1"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--projection", default=str(ROOT / "design_v126" / "projection.npy"))
    parser.add_argument("--pretrain-checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--development-result")
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = PRETRAIN_EPOCHS if args.stage == "pretrain" else 30
    print(json.dumps(run(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
