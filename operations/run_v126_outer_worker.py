"""外层常驻运维入口：一次导入科学代码，依次执行一个年份/网络种子的 A 和三个 B。"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("ordinary", "centered", "N1")


class ControlState(RuntimeError):
    def __init__(self, state, message):
        super().__init__(message)
        self.state = state


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp_{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextlib.contextmanager
def exclusive_run(lock_root, identity):
    """使用操作系统锁，进程退出自动释放；锁文件保留，避免删除产生并发竞争。"""
    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True)
    handle = (lock_root / f"{identity}.lock").open("a+b")
    if handle.tell() == 0:
        handle.write(b" ")
        handle.flush()
    handle.seek(0)
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as error:
            raise ControlState("WAITING_RESOURCE", "相同年份/网络种子的 worker 已占用运行锁") from error
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def development_contract(root):
    """三支全部通过独立 v2 验收才放行；不读取开发审计标签。"""
    result = {}
    for stage in STAGES:
        directory = Path(root) / stage
        approval_path = directory / "acceptance_integrity_v2.json"
        result_path, input_path = directory / "stage_result.json", directory / "input_receipt.json"
        if not all(p.is_file() for p in (approval_path, result_path, input_path)):
            raise ControlState("WAITING_DEPENDENCY", "2019 三个 B 分支的独立验收尚未全部完成")
        approval, completed = read(approval_path), read(result_path)
        epoch, observed = completed.get("best_epoch"), completed.get("completed_epoch")
        if (approval.get("state") != "PASS_DEVELOPMENT_EXECUTION" or approval.get("integrity_version") != 2
                or approval.get("stage") != stage or approval.get("initial_model_and_optimizer_inheritance") is not True
                or approval.get("downstream_state") != "BRANCH_READY"
                or completed.get("state") != "COMPLETED" or completed.get("stage") != stage
                or completed.get("outer_year") != 2019 or completed.get("seed") != 1009):
            raise ControlState("WAITING_DEPENDENCY", "开发执行/继承/独立验收条件未满足")
        if (type(epoch) is not int or type(observed) is not int or not 0 <= epoch <= observed <= 30
                or approval.get("best_epoch") != epoch or approval.get("completed_epoch") != observed
                or not (observed == 30 or (observed >= 5 and completed["stopping"].get("stop") is True
                                          and completed["stopping"].get("no_improvement", 0) >= 5))
                or completed["stopping"].get("best_epoch") != epoch):
            raise ControlState("PLAN_REVISION_REQUIRED", "开发停止轮次不符合已审核规则")
        if stage == "N1" and epoch == 0:
            raise ControlState("PLAN_REVISION_REQUIRED", "N1 开发停止点为 0，停止外层扩展")
        checkpoint = directory / f"checkpoint_{epoch:03d}.pt"
        if (digest(result_path) != approval["result_sha256"] or digest(input_path) != approval["input_receipt_sha256"]
                or digest(checkpoint) != approval["checkpoint_sha256"][str(epoch)]):
            raise ControlState("PLAN_REVISION_REQUIRED", "开发验收与实际停止点/检查点文件散列不一致")
        result[stage] = {"E_m": epoch, "result_path": str(result_path.resolve()),
                         "result_sha256": digest(result_path), "approval_sha256": digest(approval_path),
                         "best_checkpoint_sha256": digest(checkpoint), "A_checkpoint_sha256": approval["A_checkpoint_sha256"]}
    if len({row["A_checkpoint_sha256"] for row in result.values()}) != 1:
        raise ControlState("PLAN_REVISION_REQUIRED", "开发分支未继承同一个 A")
    return result


def import_science(code_root):
    code_root = Path(code_root).resolve()
    sys.path.insert(0, str(code_root))
    trainer = importlib.import_module("analysis_v126.train")
    source_module = importlib.import_module("analysis_v126.outer_predict")
    if (Path(trainer.__file__).resolve() != code_root / "analysis_v126/train.py"
            or Path(source_module.__file__).resolve() != code_root / "analysis_v126/outer_predict.py"):
        raise ControlState("PLAN_REVISION_REQUIRED", "当前进程已载入其它科学代码目录")
    return trainer, source_module.SourceContract


def check_checkpoint(path, expected, stage, year, seed, reader, *, complete=False):
    """读取一次检查点；科学身份不符不可退回旧点，损坏文件交由调用者保存后恢复。"""
    path = Path(path)
    payload = reader(path)
    if (payload.get("format") != "v1.26" or payload.get("stage") != stage
            or payload.get("outer_year") != year or payload.get("seed") != seed
            or payload.get("steps_per_epoch") != 100 or payload.get("scientific_files_sha256") != expected
            or not all(key in payload for key in ("model", "optimizer", "rng"))):
        raise ControlState("PLAN_REVISION_REQUIRED", "已有检查点的科学输入或运行身份不符")
    epoch = payload["epoch"]
    if type(epoch) is not int or path.name != f"checkpoint_{epoch:03d}.pt":
        raise ControlState("PLAN_REVISION_REQUIRED", "检查点轮次和文件名不一致")
    sidecar = path.with_suffix(".state.json")
    if sidecar.exists():
        if read(sidecar).get("checkpoint_sha256") != digest(path):
            raise OSError("检查点与收尾散列不一致")
    elif complete:
        raise OSError("最终检查点尚未收尾")
    return epoch


def select_resume(directory, expected, stage, year, seed, epochs, reader):
    """最新完整可读检查点优先；仅对读取/散列损坏保存隔离副本，不改变科学设置。"""
    directory = Path(directory)
    completed_path = directory / "stage_result.json"
    if completed_path.exists() and read(completed_path).get("state") == "COMPLETED":
        completed = read(completed_path)
        if any(completed.get(key) != value for key, value in {
            "stage": stage, "outer_year": year, "seed": seed, "completed_epoch": epochs, "best_epoch": epochs,
        }.items()):
            raise ControlState("PLAN_REVISION_REQUIRED", "已完成产物与固定训练轮次或身份不一致")
        final = directory / f"checkpoint_{epochs:03d}.pt"
        check_checkpoint(final, expected, stage, year, seed, reader, complete=True)
        return "COMPLETED", final, []
    damaged = []
    for path in sorted(directory.glob("checkpoint_*.pt"), reverse=True):
        try:
            epoch = check_checkpoint(path, expected, stage, year, seed, reader)
        except ControlState:
            raise
        except (OSError, EOFError, ValueError, RuntimeError, pickle.UnpicklingError) as error:
            stamp = time.time_ns()
            saved = []
            for original in (path, path.with_suffix(".state.json")):
                if original.exists():
                    preserved = original.with_name(original.name + f".invalid_{stamp}")
                    original.replace(preserved)
                    saved.append(str(preserved))
            damaged.append({"state": "RETRYABLE_INFRA", "reason": str(error), "preserved": saved})
            write(directory / f"recovery_{stamp}.json", damaged[-1])
            continue
        if epoch > epochs:
            raise ControlState("PLAN_REVISION_REQUIRED", "已有检查点超过固定 E_m")
        return "RESUME", path, damaged
    return "NEW", None, damaged


def run_worker(args, *, trainer=None, source_factory=None, checkpoint_reader=None):
    if args.year not in range(2020, 2025) or args.seed not in (1009, 2027, 3037):
        raise ControlState("PLAN_REVISION_REQUIRED", "年份或网络种子不在当前外层清单内")
    if args.threads < 1 or args.chunk_size < 1:
        raise ValueError("线程及分块大小必须为正")
    development = development_contract(args.development_root)
    identity = f"v126_y{args.year}_t{args.seed}"
    run_root = Path(args.output_root).resolve() / f"y{args.year}" / f"s{args.seed}"
    with exclusive_run(args.lock_root, identity):
        if trainer is None:
            trainer, source_factory = import_science(args.code_root)
        source = source_factory(args.data_dir, args.year, args.r2_sources)
        if checkpoint_reader is None:
            checkpoint_reader = lambda path: trainer.torch.load(path, map_location="cpu", weights_only=False)
        projection = Path(args.projection or Path(args.code_root) / "design_v126/projection.npy").resolve()
        hashes = {stage: trainer.scientific_hashes(source.data_dir, projection, stage) for stage in ("pretrain", *STAGES)}
        for relative in hashes["ordinary"]:
            if relative.startswith("roles/"):
                if digest(source.verify(relative)) != hashes["ordinary"][relative]:
                    raise ControlState("PLAN_REVISION_REQUIRED", "支持来源与训练科学输入不一致")
        for stage in STAGES:
            hashes[stage]["development_stage_result.json"] = development[stage]["result_sha256"]
        binding = {"identity": identity, "year": args.year, "seed": args.seed,
                   "scientific_files_sha256": hashes, "development": development,
                   "execution_sources_sha256": source.registry_sha256}
        # 路径不参与科学等价判定；文件散列必须相同，允许经核验的远程等价复制。
        for row in binding["development"].values():
            row.pop("result_path", None)
        binding_path = run_root / "worker_binding.json"
        if binding_path.exists() and read(binding_path) != binding:
            raise ControlState("PLAN_REVISION_REQUIRED", "该运行目录已绑定其它科学输入，不能覆盖")
        if not binding_path.exists():
            write(binding_path, binding)
        attempt = f"attempt_{time.time_ns()}"
        attempt_root = run_root / "worker_attempts" / attempt
        summary = {"state": "RUNNING", "identity": identity, "pid": os.getpid(), "host": socket.gethostname(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "stages": {},
                   "device": args.device, "threads": args.threads, "chunk_size": args.chunk_size,
                   "operations_files_sha256": {"run_v126_outer_worker.py": digest(Path(__file__))},
                   "algorithm_changes": False}
        write(attempt_root / "status.json", summary)
        try:
            for stage in ("pretrain", *STAGES):
                epochs = 20 if stage == "pretrain" else development[stage]["E_m"]
                directory = run_root / stage
                state, checkpoint, preserved = select_resume(directory, hashes[stage], stage, args.year, args.seed, epochs, checkpoint_reader)
                stage_status = {"state": state, "epochs": epochs, "preserved": preserved}
                summary["stages"][stage] = stage_status
                write(attempt_root / "status.json", summary)
                if state != "COMPLETED":
                    parameters = SimpleNamespace(data_dir=str(source.data_dir), output=str(directory), stage=stage, epochs=epochs,
                        seed=args.seed, device=args.device, threads=args.threads, chunk_size=args.chunk_size,
                        projection=str(projection), resume_checkpoint=str(checkpoint) if checkpoint else None,
                        pretrain_checkpoint=str(run_root / "pretrain/checkpoint_020.pt") if stage != "pretrain" and checkpoint is None else None,
                        development_result=str(Path(args.development_root) / stage / "stage_result.json") if stage != "pretrain" else None)
                    write(attempt_root / f"{stage}_invocation.json", vars(parameters))
                    with (attempt_root / f"{stage}.log").open("x", encoding="utf-8", buffering=1) as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            trainer.run(parameters)
                    state, checkpoint, _ = select_resume(directory, hashes[stage], stage, args.year, args.seed, epochs, checkpoint_reader)
                    if state != "COMPLETED":
                        raise ControlState("RETRYABLE_INFRA", "训练返回后尚无完整阶段产物")
                stage_status.update(state="COMPLETED", exit_code=0, checkpoint_sha256=digest(checkpoint),
                                    result_sha256=digest(directory / "stage_result.json"))
                write(attempt_root / "status.json", summary)
                gc.collect()
            summary["state"] = "COMPLETED"
            write(attempt_root / "status.json", summary)
            if not (run_root / "worker_result.json").exists():
                write(run_root / "worker_result.json", summary)
            return summary
        except BaseException as error:
            memory_failure = type(error).__name__ == "OutOfMemoryError" or (
                isinstance(error, RuntimeError) and any(message in str(error).lower()
                for message in ("out of memory", "can't allocate memory", "cannot allocate memory")))
            summary["state"] = error.state if isinstance(error, ControlState) else (
                "RETRYABLE_INFRA" if memory_failure or isinstance(error, (OSError, EOFError, KeyboardInterrupt)) else "PLAN_REVISION_REQUIRED")
            summary["error_type"], summary["error"] = type(error).__name__, str(error)
            summary["traceback"] = traceback.format_exc()
            if "stage" in locals():
                summary["stages"].setdefault(stage, {}).update(state=summary["state"], exit_code=1)
            write(attempt_root / "status.json", summary)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--lock-root", default=str(ROOT / "runs/v126/outer_worker_locks"))
    parser.add_argument("--projection")
    parser.add_argument("--r2-sources")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    try:
        result = run_worker(args)
        print(json.dumps({"state": result["state"], "identity": result["identity"]}, ensure_ascii=False))
    except ControlState as error:
        print(json.dumps({"state": error.state, "reason": str(error)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
