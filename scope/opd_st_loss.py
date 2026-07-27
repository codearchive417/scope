"""Token-level Wasserstein Distance loss for SCOPE (OPD+ST)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sinkhorn_wasserstein(
    p_source: torch.Tensor,
    p_target: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float,
    sinkhorn_iters: int,
    eps: float,
) -> torch.Tensor:
    """Entropic OT via Sinkhorn. Gradients flow through p_source only if p_target/cost are detached."""
    orig_dtype = p_source.dtype
    p_source = p_source.float()
    p_target = p_target.float()
    cost = cost.float()

    epsilon = max(float(epsilon), eps)
    kernel = torch.exp(-cost / epsilon).clamp_min(eps)

    u = torch.ones_like(p_source)
    v = torch.ones_like(p_target)
    for _ in range(sinkhorn_iters):
        u = p_source / (kernel @ v + eps)
        v = p_target / (kernel.t() @ u + eps)

    transport = u.unsqueeze(1) * kernel * v.unsqueeze(0)
    wd = (transport * cost).sum()
    return wd.to(orig_dtype)


def _compute_wd_at_position(
    student_logits_t: torch.Tensor,
    teacher_logits_t: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float,
    topk: int,
    epsilon: float,
    sinkhorn_iters: int,
    eps: float,
) -> torch.Tensor | None:
    temperature = max(float(temperature), eps)

    log_p_student = F.log_softmax(student_logits_t / temperature, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits_t.detach() / temperature, dim=-1)
    p_student = log_p_student.exp()
    p_teacher = log_p_teacher.exp().detach()

    vocab_size = p_student.size(-1)
    k = min(int(topk), vocab_size)
    if k <= 0:
        return None

    _, topk_teacher = torch.topk(p_teacher.detach(), k=k)
    _, topk_student = torch.topk(p_student.detach(), k=k)
    support = torch.unique(torch.cat([topk_teacher, topk_student], dim=0))
    if support.numel() == 0:
        return None

    p_student_sub = p_student[support]
    p_teacher_sub = p_teacher[support].detach()
    student_mass = p_student_sub.sum()
    teacher_mass = p_teacher_sub.sum()
    if student_mass.detach().item() <= eps or teacher_mass.detach().item() <= eps:
        return None

    p_student_sub = p_student_sub / (student_mass + eps)
    p_teacher_sub = p_teacher_sub / (teacher_mass + eps)

    with torch.no_grad():
        embeddings = F.normalize(
            embedding_weight[support].float(),
            p=2,
            dim=-1,
            eps=eps,
        )
        cost = 1.0 - embeddings @ embeddings.t()
        cost = cost.clamp_min(0.0)

    cost = cost.to(dtype=p_student_sub.dtype, device=p_student_sub.device).detach()
    return _sinkhorn_wasserstein(
        p_student_sub,
        p_teacher_sub,
        cost,
        epsilon=epsilon,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )


def compute_opd_st_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 1.0,
    topk: int = 32,
    lambda_wd: float = 0.1,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 20,
    wd_interval: int = 4,
    eps: float = 1e-8,
    multiply_temp_squared: bool = True,
) -> dict[str, torch.Tensor | int]:
    """SCOPE loss = reverse KL + lambda_wd * token-level Sinkhorn WD."""
    teacher_logits = teacher_logits.detach()
    temperature = max(float(temperature), eps)
    mask = loss_mask if loss_mask.dtype == torch.bool else loss_mask != 0

    log_p_student = F.log_softmax(student_logits / temperature, dim=-1)
    log_p_teacher = F.log_softmax(teacher_logits / temperature, dim=-1)
    p_student = log_p_student.exp()
    rkl_per_token = (p_student * (log_p_student - log_p_teacher)).sum(dim=-1)
    if multiply_temp_squared:
        rkl_per_token = rkl_per_token * (temperature**2)

    rkl_loss = rkl_per_token[mask].mean() if mask.any() else rkl_per_token.sum() * 0.0

    batch_size = student_logits.shape[0]
    wd_interval = max(1, int(wd_interval))
    wd_values: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        valid_positions = mask[batch_idx].nonzero(as_tuple=True)[0]
        if valid_positions.numel() == 0:
            continue
        for token_idx in valid_positions[::wd_interval]:
            wd_value = _compute_wd_at_position(
                student_logits[batch_idx, token_idx],
                teacher_logits[batch_idx, token_idx],
                embedding_weight,
                temperature=temperature,
                topk=topk,
                epsilon=epsilon,
                sinkhorn_iters=sinkhorn_iters,
                eps=eps,
            )
            if wd_value is not None and torch.isfinite(wd_value.detach()).item():
                wd_values.append(wd_value)

    num_wd_positions = len(wd_values)
    num_valid_loss_tokens = int(mask.sum().detach().item())
    wd_loss = torch.stack(wd_values).mean() if num_wd_positions > 0 else student_logits.sum() * 0.0
    total_loss = rkl_loss + float(lambda_wd) * wd_loss

    return {
        "loss": total_loss,
        "rkl_loss": rkl_loss.detach(),
        "wd_loss": wd_loss.detach(),
        "num_wd_positions": num_wd_positions,
        "num_valid_loss_tokens": num_valid_loss_tokens,
    }
