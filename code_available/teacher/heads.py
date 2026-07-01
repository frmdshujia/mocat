# [Paper §Method] Per-task 2-layer MLP prediction heads (regression + classification).

"""
heads.py — task head for the CMR G3 teacher.

TaskHead is lifted verbatim from the original
`contrastive_pretrain/models_dualtower.py` (the storage servers) and extracted into a
standalone, dependency-free module. The original `models_dualtower.py` also
pulls in RETFound (`models_vit`) and the fundus tower, none of which the CMR
teacher needs — so we keep only TaskHead here to keep `model/teacher/`
self-contained and easy to ablate.

Canonical G3 head:  n_cls=3 (prevalent_I25, history_revasc, prevalent_I21)
                    n_reg=5 (LV EDV, LV ESV, LV mass, RV EDV, RV ESV)
The checkpoint's head has 32 tensors = 3 cls + 5 reg heads × (2 Linear × {w,b})
plus none for the dropout. (3+5) heads × 2 Linears × 2 params = 32. ✓
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """Multi-task head: per-task 2-layer MLP on the pooled 768-d CLS embedding.

    forward(z) -> (cls_out, reg_out)
      cls_out: list[n_cls] of (B,) logits   (apply sigmoid for probability)
      reg_out: list[n_reg] of (B,) values    (z-scored; de-normalise with REG_NORM)
    """
    def __init__(self, in_dim: int, n_cls: int = 4, n_reg: int = 3, dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_cls)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_reg)
        ])

    def forward(self, z: torch.Tensor):
        z = self.drop(z)
        cls_out = [h(z).squeeze(-1) for h in self.cls_heads]
        reg_out = [h(z).squeeze(-1) for h in self.reg_heads]
        return cls_out, reg_out


class GatedTaskHead(nn.Module):
    """④ Target-gated view-expert head.

    Per task, a learned gate weights three view representations
    {z_cls (joint), expert_lax(z_lax), expert_sax(z_sax)} via a softmax, so each
    metric can rely on the view that physically carries it (GLS<-LAX longitudinal,
    GCS/GRS<-SAX circumferential/radial). Drop-in for TaskHead: forward returns
    (cls_out, reg_out) with the SAME list structure so task_loss is unchanged.
    """
    def __init__(self, in_dim: int, n_cls: int = 7, n_reg: int = 26, dropout: float = 0.3):
        super().__init__()
        self.n_cls = n_cls; self.n_reg = n_reg; self.n_task = n_cls + n_reg
        self.drop = nn.Dropout(dropout)
        self.expert_lax = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, in_dim))
        self.expert_sax = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Linear(in_dim, in_dim))
        self.gate = nn.Linear(3 * in_dim, self.n_task * 3)         # per-task logits over 3 views
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(self.n_task)
        ])

    def forward(self, z_cls: torch.Tensor, z_lax: torch.Tensor, z_sax: torch.Tensor):
        # ③+④ combo: when ③ (deform) is on, the encoder forces z_cls→float32 while z_lax/z_sax
        # follow autocast (half). Cast expert outputs + gate back to z_cls.dtype so the stack/cat/
        # einsum below never mix Half and Float. (No-op when all three already share a dtype, e.g.
        # ④ alone under autocast = all half, or any path on CPU = all float.)
        d = z_cls.dtype
        el = self.expert_lax(z_lax).to(d); es = self.expert_sax(z_sax).to(d)
        views = torch.stack([z_cls, el, es], dim=1)               # (B, 3, D)
        g = self.gate(torch.cat([z_cls, el, es], dim=-1)).to(d)   # (B, n_task*3)
        g = g.view(-1, self.n_task, 3).softmax(dim=-1)            # (B, n_task, 3)
        z_t = torch.einsum('btv,bvd->btd', g, views)              # (B, n_task, D)
        z_t = self.drop(z_t)
        outs = [self.task_heads[t](z_t[:, t]).squeeze(-1) for t in range(self.n_task)]
        return outs[:self.n_cls], outs[self.n_cls:]
