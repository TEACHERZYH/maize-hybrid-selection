"""常驻 worker 仅模拟训练接口，验证依赖、锁和恢复；不导入或训练真实网络。"""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import run_v126_outer_worker as worker


class FakeSource:
    def __init__(self, data_dir, year, registry_path=None):
        self.data_dir = Path(data_dir)
        self.registry_sha256 = None
        if year == 2024:
            registry = worker.read(registry_path)
            if self.data_dir.resolve() != Path(registry["required_2024_data_dir"]).resolve():
                raise ValueError("2024 旧目录被拒绝")
            self.registry_sha256 = worker.digest(registry_path)

    def verify(self, relative):
        return self.data_dir / relative


class FakeTrainer:
    def __init__(self, fail_stage=None, corrupt=False):
        self.calls, self.fail_stage, self.corrupt = [], fail_stage, corrupt

    def scientific_hashes(self, data_dir, projection, stage):
        result = {"history.npz": worker.digest(Path(data_dir) / "history.npz"),
                  "projection.npy": worker.digest(projection), "analysis_v126/train.py": "frozen-science"}
        if stage != "pretrain":
            result["target_features.npz"] = worker.digest(Path(data_dir) / "target_features.npz")
        return result

    def checkpoint(self, args, epoch):
        hashes = self.scientific_hashes(args.data_dir, args.projection, args.stage)
        if args.development_result:
            hashes["development_stage_result.json"] = worker.digest(args.development_result)
        path = Path(args.output) / f"checkpoint_{epoch:03d}.pt"
        worker.write(path, {"format": "v1.26", "stage": args.stage, "outer_year": 2020, "seed": args.seed,
                            "epoch": epoch, "steps_per_epoch": 100, "scientific_files_sha256": hashes,
                            "model": {"fresh_instance": len(self.calls)}, "optimizer": {}, "rng": {}})
        worker.write(path.with_suffix(".state.json"), {"checkpoint_sha256": worker.digest(path)})

    def run(self, args):
        self.calls.append(copy.deepcopy(vars(args)))
        if args.stage != "pretrain" and not args.resume_checkpoint:
            inherited = worker.read(args.pretrain_checkpoint)
            assert inherited["stage"] == "pretrain" and inherited["epoch"] == 20
        if args.resume_checkpoint:
            assert args.pretrain_checkpoint is None
            assert worker.read(args.resume_checkpoint)["stage"] == args.stage
        else:
            self.checkpoint(args, 0)
        if args.stage == self.fail_stage:
            if self.corrupt:
                (Path(args.output) / "checkpoint_001.pt").write_text("truncated", encoding="utf-8")
            self.fail_stage = None
            raise OSError("模拟共享文件系统中断")
        self.checkpoint(args, args.epochs)
        result = {"state": "COMPLETED", "stage": args.stage, "outer_year": 2020, "seed": args.seed,
                  "completed_epoch": args.epochs, "best_epoch": args.epochs}
        worker.write(Path(args.output) / "stage_result.json", result)
        print("模拟训练日志", args.stage)
        return result


def prepare(tmp_path, centered_epoch=2):
    development = tmp_path / "development"
    for stage, epoch in (("ordinary", 2), ("centered", centered_epoch), ("N1", 1)):
        directory = development / stage
        result_path, input_path = directory / "stage_result.json", directory / "input_receipt.json"
        result = {"state": "COMPLETED", "stage": stage, "outer_year": 2019, "seed": 1009,
                  "best_epoch": epoch, "completed_epoch": epoch + 5,
                  "stopping": {"best_epoch": epoch, "stop": True, "no_improvement": 5}}
        worker.write(result_path, result)
        worker.write(input_path, {"synthetic": True})
        checkpoint = directory / f"checkpoint_{epoch:03d}.pt"
        worker.write(checkpoint, {"synthetic": "already_independently_accepted"})
        worker.write(directory / "acceptance_integrity_v2.json", {
            "state": "PASS_DEVELOPMENT_EXECUTION", "integrity_version": 2, "stage": stage,
            "best_epoch": epoch, "completed_epoch": epoch + 5, "downstream_state": "BRANCH_READY",
            "initial_model_and_optimizer_inheritance": True, "A_checkpoint_sha256": "same_A",
            "result_sha256": worker.digest(result_path), "input_receipt_sha256": worker.digest(input_path),
            "checkpoint_sha256": {str(epoch): worker.digest(checkpoint)},
        })
    data = tmp_path / "data"
    data.mkdir()
    (data / "history.npz").write_bytes(b"synthetic-history")
    (data / "target_features.npz").write_bytes(b"synthetic-target")
    projection = tmp_path / "projection.npy"
    projection.write_bytes(b"synthetic-fixed-P")
    return SimpleNamespace(year=2020, seed=1009, data_dir=str(data), output_root=str(tmp_path / "runs"),
                           code_root=str(tmp_path / "frozen"), development_root=str(development),
                           lock_root=str(tmp_path / "locks"), projection=str(projection), r2_sources=None,
                           device="cpu", threads=4, chunk_size=512)


def invoke(args, trainer):
    return worker.run_worker(args, trainer=trainer, source_factory=FakeSource, checkpoint_reader=worker.read)


def test_single_import_A_then_three_fresh_B_and_zero_epoch_control(tmp_path, monkeypatch):
    args, trainer = prepare(tmp_path, centered_epoch=0), FakeTrainer()
    imports = []
    def imported(path):
        imports.append(path)
        return trainer, FakeSource
    monkeypatch.setattr(worker, "import_science", imported)
    result = worker.run_worker(args, checkpoint_reader=worker.read)
    assert imports == [args.code_root]
    assert result["state"] == "COMPLETED"
    assert [call["stage"] for call in trainer.calls] == ["pretrain", "ordinary", "centered", "N1"]
    assert [call["epochs"] for call in trainer.calls] == [20, 2, 0, 1]
    assert len({call["pretrain_checkpoint"] for call in trainer.calls[1:]}) == 1
    assert all(call["resume_checkpoint"] is None and call["seed"] == 1009 and call["threads"] == 4 for call in trainer.calls)
    assert len({call["output"] for call in trainer.calls}) == 4
    assert len(list((Path(args.output_root) / "y2020/s1009/worker_attempts").glob("attempt_*/*.log"))) == 4


def test_all_development_branches_required_before_any_training(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    path = Path(args.development_root) / "ordinary/acceptance_integrity_v2.json"
    approval = worker.read(path)
    approval["state"] = "WAITING_RUNNING"
    worker.write(path, approval)
    with pytest.raises(worker.ControlState) as error:
        invoke(args, trainer)
    assert error.value.state == "WAITING_DEPENDENCY" and not trainer.calls


def test_N1_zero_blocks_entire_worker(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    path = Path(args.development_root) / "N1/stage_result.json"
    result = worker.read(path)
    result.update(best_epoch=0, completed_epoch=5)
    result["stopping"]["best_epoch"] = 0
    worker.write(path, result)
    approval_path = path.parent / "acceptance_integrity_v2.json"
    approval = worker.read(approval_path)
    approval.update(best_epoch=0, completed_epoch=5)
    worker.write(approval_path, approval)
    with pytest.raises(worker.ControlState) as error:
        invoke(args, trainer)
    assert error.value.state == "PLAN_REVISION_REQUIRED" and not trainer.calls


def test_duplicate_worker_refused_and_lock_released(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    identity = "v126_y2020_t1009"
    with worker.exclusive_run(args.lock_root, identity):
        with pytest.raises(worker.ControlState) as error:
            invoke(args, trainer)
        assert error.value.state == "WAITING_RESOURCE" and not trainer.calls
    assert invoke(args, trainer)["state"] == "COMPLETED"


def test_valid_completed_artifacts_skipped_and_immutable(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    invoke(args, trainer)
    root = Path(args.output_root) / "y2020/s1009"
    original = {p: worker.digest(p) for p in root.glob("*/checkpoint_*.pt")}
    original[root / "worker_result.json"] = worker.digest(root / "worker_result.json")
    assert invoke(args, trainer)["state"] == "COMPLETED"
    assert len(trainer.calls) == 4
    assert all(worker.digest(path) == value for path, value in original.items())


def test_infrastructure_resume_uses_latest_valid_checkpoint_and_preserves_damage(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer(fail_stage="centered", corrupt=True)
    with pytest.raises(OSError):
        invoke(args, trainer)
    root = Path(args.output_root) / "y2020/s1009"
    statuses = list((root / "worker_attempts").glob("attempt_*/status.json"))
    assert worker.read(statuses[0])["state"] == "RETRYABLE_INFRA"
    result = invoke(args, trainer)
    assert result["state"] == "COMPLETED"
    assert [c["stage"] for c in trainer.calls] == ["pretrain", "ordinary", "centered", "centered", "N1"]
    resumed = trainer.calls[3]
    assert resumed["resume_checkpoint"].endswith("checkpoint_000.pt") and resumed["pretrain_checkpoint"] is None
    assert len(list((root / "centered").glob("checkpoint_001.pt.invalid_*"))) == 1
    assert result["stages"]["centered"]["preserved"][0]["state"] == "RETRYABLE_INFRA"


def test_changed_scientific_inputs_never_reuse_finished_outputs(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    invoke(args, trainer)
    (Path(args.data_dir) / "history.npz").write_bytes(b"changed-science")
    with pytest.raises(worker.ControlState) as error:
        invoke(args, trainer)
    assert error.value.state == "PLAN_REVISION_REQUIRED" and len(trainer.calls) == 4


def test_wrong_checkpoint_science_cannot_fallback_to_another_point(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer(fail_stage="centered")
    with pytest.raises(OSError):
        invoke(args, trainer)
    path = Path(args.output_root) / "y2020/s1009/centered/checkpoint_000.pt"
    payload = worker.read(path)
    payload["scientific_files_sha256"]["analysis_v126/train.py"] = "other_loss"
    worker.write(path, payload)
    worker.write(path.with_suffix(".state.json"), {"checkpoint_sha256": worker.digest(path)})
    with pytest.raises(worker.ControlState) as error:
        invoke(args, trainer)
    assert error.value.state == "PLAN_REVISION_REQUIRED" and len(trainer.calls) == 3
    assert path.exists()


def test_2024_old_source_rejected_before_train(tmp_path):
    args, trainer = prepare(tmp_path), FakeTrainer()
    args.year = 2024
    registry = tmp_path / "sources.json"
    worker.write(registry, {"required_2024_data_dir": str(tmp_path / "new2024")})
    args.r2_sources = str(registry)
    with pytest.raises(ValueError):
        invoke(args, trainer)
    assert not trainer.calls
