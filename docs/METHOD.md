# Method design: SCOPE

**SCOPE** = offline trajectories + **OPD+RC** (rewrite) + frozen teacher + **OPD+ST** (loss).

## Loss

\[
\mathcal{L}_{\text{OPD+ST}}
=
\mathrm{RKL}(p_S \,\|\, p_T)
+
\lambda_{\text{wd}} \cdot
\mathrm{WD}_{\text{token}}(p_S, p_T)
\]

Implementation: `scope/opd_st_loss.py` → `compute_opd_st_loss`.

### Reverse KL

On completion tokens with `loss_mask`:

1. \(p_S=\mathrm{softmax}(z_S/T)\), \(p_T=\mathrm{softmax}(z_T/T)\) (teacher detached)
2. \(\mathrm{RKL}_t=\sum_v p_S(v)(\log p_S(v)-\log p_T(v))\)
3. Optional \(T^2\) scaling (`multiply_temp_squared=True`)
4. Mean over masked tokens → `rkl_loss`

### Token-level Sinkhorn WD

At subsampled completion positions (`wd_interval`):

1. Take student/teacher top-k mass (`wd_topk`), union support
2. Cost from L2-normalized embeddings: \(C_{ij}=1-\cos(e_i,e_j)\) (detached)
3. Entropic OT via Sinkhorn (`wd_epsilon`, `wd_sinkhorn_iters`) in fp32
4. Gradients flow through student probability mass only
5. Average over selected positions → `wd_loss`

Total: `loss = rkl_loss + lambda_wd * wd_loss`.

## Three modules

| Module | Role |
|--------|------|
| OPD+RC | Offline 8B `teacher_reasoning` → online rewrite |
| 8B teacher | Frozen teacher logits via `--teacher_model_name_or_path` |
| OPD+ST | Distillation loss: RKL + λ·token WD |

## Pipeline

```text
Offline:
  problem + GT solution --[8B]--> teacher_reasoning

Online:
  OPD+RC -> teacher privileged context
  student on-policy sample -> dual forward -> OPD+ST
```

## Design notes

- RKL keeps token-distribution alignment; WD softens near-synonym mismatches via embedding geometry.
- Top-k support makes OT tractable on large vocabularies.
- `wd_interval` reduces cost vs computing WD at every token.
- Sinkhorn runs in fp32 under bf16 training for stability.
