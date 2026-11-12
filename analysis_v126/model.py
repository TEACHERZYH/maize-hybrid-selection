"""v1.26：作者 TabM 与可微的年度共享响应更新。

第三方实现保持原样，出处和哈希见 vendor/source_receipt.json。
本模块不接受隐藏标签；调用方只传支持标签和完整候选池的特征。
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .vendor.tabm import LinearEnsemble, TabM


class ResponseTabM(nn.Module):
    """固定 3×128、16 成员的 TabM；返回未中心化的 8 维响应表示。"""

    def __init__(self, input_dim: int, seed: int, projection_seed: int = 12600):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim 必须为正整数")
        self.input_dim = input_dim
        self.seed = seed
        self.projection_seed = projection_seed
        # 仅在 CPU 初始化；恢复调用方随机流，P 使用独立固定随机源。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = TabM.make(
                n_num_features=input_dim, d_out=None, num_embeddings=None,
                n_blocks=3, d_block=128, k=16, dropout=0.0,
                activation="ReLU", arch_type="tabm",
                start_scaling_init="random-signs",
            )
            self.output = LinearEnsemble(128, 1, k=16)
        raw = np.random.default_rng(projection_seed).standard_normal((128, 8))
        q, r = np.linalg.qr(raw, mode="reduced")
        signs = np.where(np.diag(r) < 0, -1.0, 1.0)
        projection = np.ascontiguousarray(q * signs, dtype=np.float32)
        self.register_buffer("projection", torch.from_numpy(projection))

    @property
    def projection_sha256(self) -> str:
        """实际使用的 float32 投影，按 C 序小端字节计算哈希。"""
        array = self.projection.detach().cpu().numpy().astype("<f4", copy=False)
        return hashlib.sha256(array.tobytes(order="C")).hexdigest()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError("x 必须为 [组合数, input_dim]")
        if x.dtype != torch.float32:
            raise ValueError("v1.26 网络前向固定为 float32")
        h = self.encoder(x)
        base = self.output(h).squeeze(-1)
        phi = torch.tanh(h @ self.projection)
        return base, phi

    def forward_chunked(
        self, x: Tensor, chunk_size: int = 512, checkpoint_chunks: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """分块只限制前向内存；不分块求中心均值或岭系数。

        激活重算保留每一行对隐藏层的梯度，不能用 no_grad 前向替代。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须为正整数")
        if len(x) == 0:
            raise ValueError("候选池不能为空")
        outputs = []
        for chunk in x.split(chunk_size):
            if checkpoint_chunks and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self.forward, chunk, use_reentrant=False))
            else:
                outputs.append(self(chunk))
        return tuple(torch.cat(parts, dim=0) for parts in zip(*outputs))


def _index(value: Tensor, device: torch.device, name: str) -> Tensor:
    value = torch.as_tensor(value, device=device)
    if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} 必须是一维整数索引")
    return value.long()


def adapt_predict(
    base: Tensor, phi: Tensor, env_index: Tensor, support_idx: Tensor,
    support_y: Tensor, mode: str = "response", lambda_a: float = 1.0,
    lambda_w: float = 10.0,
) -> tuple[Tensor, dict[str, Any]]:
    """对完整支持 S 一次性联合拟合；每成员共享年度响应，截距按环境。

    base=[B,K]，phi=[B,K,8]，V 是全部 B 行；support_y 只含 S 的标签。
    返回 [B,K] 的 float64 预测，调用方对 K 取均值。诊断张量不带梯度。
    zero 必须传空支持；intercept 只求解截距。支持索引禁止重复。
    """
    if mode not in {"response", "intercept", "zero"}:
        raise ValueError("未知更新模式")
    if base.ndim != 2 or base.shape[0] == 0:
        raise ValueError("base 必须为非空 [B,K]")
    if phi.shape != (*base.shape, 8) or phi.device != base.device:
        raise ValueError("phi 必须为同设备上的 [B,K,8]")
    if not (np.isfinite(lambda_a) and np.isfinite(lambda_w)) or min(lambda_a, lambda_w) <= 0:
        raise ValueError("岭惩罚必须有限且为正")
    if not (torch.isfinite(base).all() and torch.isfinite(phi).all()):
        raise ValueError("输入包含非有限预测或表示")
    env_index = _index(env_index, base.device, "env_index")
    support_idx = _index(support_idx, base.device, "support_idx")
    if len(env_index) != len(base):
        raise ValueError("环境索引长度与候选池不一致")
    if len(support_idx) and (support_idx.min() < 0 or support_idx.max() >= len(base)):
        raise ValueError("支持索引越界")
    if len(torch.unique(support_idx)) != len(support_idx):
        raise ValueError("支持索引重复；S0 只能进入一次联合求解")
    support_y = torch.as_tensor(support_y, device=base.device, dtype=torch.float64)
    if support_y.ndim != 1 or len(support_y) != len(support_idx):
        raise ValueError("只允许传入与 S 顺序一致的一维支持标签")
    if not torch.isfinite(support_y).all():
        raise ValueError("支持标签包含非有限值")
    if mode == "zero" and len(support_idx):
        raise ValueError("zero 模式须传空支持，不观察 S0")
    env_values, inv, counts = torch.unique(env_index, sorted=True, return_inverse=True, return_counts=True)
    n_env, n_members = len(env_values), base.shape[1]
    p64 = phi.double()
    sums = p64.new_zeros((n_env, n_members, 8)).index_add(0, inv, p64)
    psi = p64 - (sums / counts[:, None, None])[inv]
    base64 = base.double()
    dimensions = n_env + (8 if mode == "response" else 0)
    diag: dict[str, Any] = {
        "mode": mode, "support_count": len(support_idx), "environment_values": env_values.detach(),
        "environment_candidate_count": counts.detach(),
        "tanh_saturation_fraction": (phi.detach().abs() >= 0.99).double().mean(),
    }
    if len(support_idx) == 0:
        diag.update({
            "beta": base64.new_zeros((n_members, dimensions)),
            "ridge_condition": base64.new_full((n_members,), max(lambda_a, lambda_w) / min(lambda_a, lambda_w) if mode == "response" else 1.),
            "response_rank": torch.zeros(n_members, dtype=torch.long, device=base.device),
            "response_singular_min": base64.new_zeros(n_members),
            "response_singular_max": base64.new_zeros(n_members),
            "effective_df": base64.new_zeros(n_members),
            "intercept_rms": base64.new_zeros(n_members),
            "within_environment_update_rms": base64.new_zeros(n_members),
        })
        return base64, diag
    support_env = inv[support_idx]
    onehot = torch.nn.functional.one_hot(support_env, n_env).double()
    z = onehot.unsqueeze(0).expand(n_members, -1, -1)
    if mode == "response":
        z = torch.cat((z, psi[support_idx].transpose(0, 1)), dim=-1)
    penalty = base64.new_full((dimensions,), lambda_a)
    if mode == "response":
        penalty[n_env:] = lambda_w
    gram = z.transpose(-1, -2) @ z
    system = gram + torch.diag(penalty).unsqueeze(0)
    residual = (support_y[:, None] - base64[support_idx]).T.unsqueeze(-1)
    chol = torch.linalg.cholesky(system)
    beta = torch.cholesky_solve(z.transpose(-1, -2) @ residual, chol).squeeze(-1)
    intercept = beta[:, :n_env].T[inv]
    response = (psi * beta[:, None, n_env:].transpose(0, 1)).sum(-1) if mode == "response" else torch.zeros_like(base64)
    prediction = base64 + intercept + response
    # 诊断只读取本轮张量，不改变解或训练梯度。S 中去除环境均值后查可识别秩。
    with torch.no_grad():
        sp = psi[support_idx].detach()
        support_count = torch.bincount(support_env, minlength=n_env)
        support_sums = sp.new_zeros((n_env, n_members, 8)).index_add(0, support_env, sp)
        centered_support = sp - (support_sums / support_count.clamp_min(1)[:, None, None])[support_env]
        singular = torch.linalg.svdvals(centered_support.transpose(0, 1))
        tolerance = max(len(support_idx), 8) * torch.finfo(torch.float64).eps * singular[:, :1]
        effective_df = torch.diagonal(torch.cholesky_solve(gram.detach(), chol.detach()), dim1=-2, dim2=-1).sum(-1)
        diag.update({
            "beta": beta.detach(), "ridge_condition": torch.linalg.cond(system.detach()),
            "response_rank": (singular > tolerance).sum(-1),
            "response_singular_min": singular[:, -1] if len(support_idx) >= 8 else singular.new_zeros(n_members),
            "response_singular_max": singular[:, 0], "effective_df": effective_df,
            "intercept_rms": beta[:, :n_env].detach().square().mean(-1).sqrt(),
            "within_environment_update_rms": response.detach().square().mean(0).sqrt(),
        })
    return prediction, diag
