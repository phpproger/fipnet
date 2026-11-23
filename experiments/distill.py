"""
Two-half distillation for the resonance window LM (FIP), exploiting the
architecture's natural split at h[:, -1]:

  compressor (steps 1-5): byte-harmonic stem -> lift -> PE -> CausalAttention
          -> mix BlockV2 -> first BlockV2   ==>  u_S = h[:, -1] (d_S-dim)
  decoder stack (steps 6-7): remaining BlockV2 stack + low-rank phase head
          + resonance codebook  ==>  prediction from ONE vector.

No 1:1 layer binding to the teacher is needed: the teacher may be any depth.

Stage 0 (seconds, data-free): align the codebook c_v with the TEACHER head
  geometry - fit ||C~C~^T - G_T||_F^2 + Welch repulsion (G_T = pairwise
  cosine matrix of the teacher's head rows). The 32-dim target then agrees
  with the teacher's class structure instead of a plain sphere packing.

Stage 1 (real windows): distill the compressor. P (d_S x d_T) is a
  semi-orthogonal projector fitted by SVD over teacher last-position states
  u_T = h_T[:, -1] (pre-head, post-final-norm). Train stem + attention + mix
  + first blocks on 1 - cos(u_S, u_T P^T). Steps 6-7 are not in the graph.
  The compressor is variance-preserving by construction (cos^2+sin^2 = 1),
  so u_S scale is stable.

Stage 2 ("fry", millions of vectors, pure GEMM, no windows, no teacher
  forward): the teacher target on a synthetic vector is ANALYTIC:
      L_T(u~) = (u~ P) W_head^T,  p_T = softmax(L_T / T_kd).
  1. Buffer B_S of ~100k real u_S from the frozen compressor.
  2. Spherical mixup: u~ = r * (a u_a + (1-a) u_b + xi) / ||...||,
     a ~ U(-0.1, 1.1) - stay on the energy sphere (r = sqrt(d_S) default).
  3. Student stack + head -> z_hat (K-dim), resonance logits
     s = tau * <z_hat, c_v> / sqrt(|z_hat|^2 + tau0^2).
  4. Losses:
       softvec: z*_target = sum_v p_T,v c_v (O(K), perfectly smooth gradient
         through the narrow head - bypasses the O(V*K) softmax entirely),
         minimize 1 - cos(z_hat, z*_target);
       kl: reverse KL(p_S || p_T) on resonance logits.
  5. Anti-collapse anchors (depth 14+ safe):
       - empirical variance preservation per block: (Var(out)-Var(hin))^2;
       - optional Newton-Schulz re-orthogonalization of channel W every
         50 steps ((3/2)W - (1/2) W W^T W).

Polish (end-to-end): small lr, reverse KL vs real teacher logits + CE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.attention_v3 import fip_wave, GAMMA, BETA


# ---------------------------------------------------------------- projection

def fit_task_projection(teacher_head_w, d_S):
    """P (d_S, d_T), rows = top right-singular directions of W_head_T:
    the d_S channels that actually reach the logits. Skips the rogue
    dimensions / null-space of the head that plain state-SVD wastes
    student channels on."""
    U, S, Vh = torch.linalg.svd(teacher_head_w.float(), full_matrices=False)
    P = Vh[:d_S].contiguous()                    # (d_S, d_T), rows orthonormal
    return P


def fit_projection(teacher, u_T, d_S):
    """P (d_S, d_T) with orthonormal rows, from SVD of teacher last-position
    states u_T (N, d_T)."""
    U, S, Vh = torch.linalg.svd(u_T.float(), full_matrices=False)
    P = Vh[:d_S].t().contiguous()               # (d_T, d_S), rows orthonormal
    err = (P.t() @ P - torch.eye(P.shape[1], device=P.device)).abs().max().item()
    assert err < 1e-2, f"projection not semi-orthogonal (err {err:.2e})"
    return P.t().contiguous()                   # (d_S, d_T)


def distill_loss(u, v, eps=1e-6, lam_norm=0.0):
    """1 - cos(u, v) [+ lam*(ln||u||-ln||v||)^2]; flats must share dim."""
    u = u.reshape(-1, u.shape[-1]).float()
    v = v.reshape(-1, v.shape[-1]).float()
    loss = (1.0 - F.cosine_similarity(u, v, dim=1, eps=eps)).mean()
    if lam_norm > 0:
        lu = u.norm(dim=1).clamp_min(eps).log()
        lv = v.norm(dim=1).clamp_min(eps).log()
        loss = loss + lam_norm * ((lu - lv) ** 2).mean()
    return loss


# ---------------------------------------------------------------- Stage 0

def align_codebook(student, teacher_head_w, steps=1500, lr=0.02,
                   lam_welch=1.0, mu_max=0.45, device="cuda"):
    """Fit the student codebook to the teacher head Gram matrix (data-free).

    teacher_head_w: (V, d_T) head rows (untied Linear.weight or tied embed).
    Loss: ||C~C~^T - G_T||_F^2 + lam * sum_offdiag relu(cos - mu_max)^2.
    Returns the aligned codebook; the decoder cache is updated in place.
    """
    dec = student.decoder
    C = dec.code_cache if isinstance(dec.code_cache, nn.Parameter) \
        else None
    if C is None:                                # promote buffer -> param
        dec.codes()
        C = dec.code_cache
    Wn = F.normalize(teacher_head_w.to(device).float(), dim=1)
    G_T = Wn @ Wn.t()                            # (V, V)
    off_mask = ~torch.eye(G_T.shape[0], dtype=torch.bool, device=device)
    opt = torch.optim.Adam([C], lr=lr)
    for _ in range(steps):
        cn = C / C.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        G_S = cn @ cn.t()
        loss = ((G_S - G_T) ** 2).mean() \
            + lam_welch * F.relu(G_S[off_mask] - mu_max).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        C.copy_(C / C.norm(dim=-1, keepdim=True))    # unit rows for resonance
    print(f"  stage0: gram loss {((cn @ cn.t() - G_T) ** 2).mean().item():.4f}")
    return C.detach()


# ---------------------------------------------------------------- compressor

def stem_forward(student, tokens):
    """Steps 1-3: byte harmonics -> lift -> phi (no PE, no attention)."""
    m = student
    z = m.bytes_mat[tokens].float().unsqueeze(-1) * m.omega.unsqueeze(0) \
        + m.theta.unsqueeze(0)
    code = fip_wave(z, GAMMA, BETA).sum(2)           # (B, W, d_code)
    return fip_wave(F.linear(code, m.lift.weight), GAMMA, BETA)


def compressor_forward(student, tokens, n_comp_blocks=1):
    """Steps 1-5: stem + PE + interleaved layers [attn_i+mix_i, block_i] for
    i < n_comp_blocks; cut h[:, -1]. Mirrors AttLM.features order exactly.
    Requires n_comp_blocks >= n_attn (all attention lives in the compressor;
    the fry stack is pointwise-only)."""
    m = student
    h = stem_forward(student, tokens)
    if m.pos_mode == "harmonic":
        h = h + m._pe(tokens.shape[1], tokens.device)
    for i in range(n_comp_blocks):
        if i < len(m.attn_layers):
            h = m.mix_blocks[i](m.attn_layers[i](h))
        h = m.blocks[i](h)
    return h[:, -1]                                  # (B, d_S) window state


def stack_forward(student, u_S, n_comp_blocks=1):
    """Steps 6-7: remaining BlockV2 stack + phase head -> z_hat (K-dim)."""
    for i in range(n_comp_blocks, len(student.blocks)):
        u_S = student.blocks[i](u_S)
    return student.decoder.head(u_S)


def collect_state_buffer(student, data, target=100_000, batch=64,
                         n_comp_blocks=1, device="cuda"):
    """Stage 2 buffer: real u_S = h[:, -1] from the frozen compressor."""
    W = student.window
    chunks = []
    while sum(c.shape[0] for c in chunks) < target:
        pos = torch.randint(0, data.shape[0] - W - 1, (batch,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        with torch.no_grad():
            chunks.append(compressor_forward(student, data[idx].long(),
                                             n_comp_blocks))
    return torch.cat(chunks)[:target].float()


class SphericalMixup:
    """u~ = r * (a u_a + (1-a) u_b + xi) / ||...||; a ~ U(-0.1, 1.1)."""
    def __init__(self, buffer, radius="empirical", noise=0.05):
        self.buf = buffer
        self.d = buffer.shape[1]
        self.radius = radius
        self.noise = noise

    def sample(self, b, device):
        idx = torch.randint(0, self.buf.shape[0], (b,), device=device)
        j = torch.randint(0, self.buf.shape[0], (b,), device=device)
        alpha = torch.empty(b, 1, device=device).uniform_(-0.1, 1.1)
        y = alpha * self.buf[idx].to(device) + (1 - alpha) * self.buf[j].to(device)
        y = y + torch.randn_like(y) * (self.noise * self.buf.std().to(device))
        r = math.sqrt(self.d) if self.radius == "fixed" \
            else self.buf[idx].norm(dim=1, keepdim=True).to(device)
        return r * y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def teacher_logits_analytic(u_syn, P, teacher_head_w, proj_norm=None):
    """L_T(u~) = (u~ P) W_head^T - the teacher is never run. When proj_norm
    is given, the lifted vector is rescaled to the teacher's NATIVE projected
    norm: the cosine-trained student lives at its own scale (u_S ~ 6.5+/-3.4),
    the teacher head was trained on RMS-normed states (~sqrt(d_T)) - mixing
    the two scales would silently change the softmax temperature of p_T."""
    u_T = u_syn @ P
    if proj_norm is not None:
        u_T = F.normalize(u_T, dim=-1) * proj_norm
    return u_T @ teacher_head_w.t().float()


# ---------------------------------------------------------------- stages

class ResonanceDistiller:
    def __init__(self, teacher, student, device="cuda",
                 n_comp_blocks=1, lam_norm=0.0, seed=0):
        self.teacher, self.student = teacher, student
        self.device, self.n_comp_blocks = device, n_comp_blocks
        self.lam_norm = lam_norm
        torch.manual_seed(seed)
        self.P = None
        self.head_w = None

    def real_windows(self, data, count):
        W = self.student.window
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=self.device)
        off = torch.arange(W, device=device if False else self.device)
        return data[pos.unsqueeze(1) + off.unsqueeze(0)].long()

    def real_states(self, data, count, sub=2048):
        """Real compressor states in sub-batches: the 8192-window forward in
        one shot OOMs the 4GB card even under no_grad (attention scores are
        (8192, 64, 64))."""
        with torch.no_grad():
            xb = self.real_windows(data, count)
            parts = []
            for i0 in range(0, xb.shape[0], sub):
                parts.append(compressor_forward(self.student, xb[i0:i0 + sub],
                                                self.n_comp_blocks))
        return torch.cat(parts).detach()

    # ---- Stage 1 -------------------------------------------------------
    def distill_compressor(self, data, steps=3000, lr=1e-3, batch=64):
        """Fit task-aware P; train stem+attn+mix+first blocks to match
        u_T P^T; freeze the compressor.

        Scale contract (two independent branches):
          * Stage 1 trains ANGLE ONLY (pure cosine, lam_norm=0): the
            variance-preserving phi-blocks cap ||u_S|| at ~6.5-11.3 and
            must never be pushed toward the teacher's projected scale
            (~30) - the student keeps its native isometric life.
          * Stage 2 feeds the stack the student's NATIVE radii (pair-
            interpolated mixup + real anchors); the teacher target branch
            runs under no_grad and rescales a COPY to the teacher's native
            projected norm (teacher_proj_norm ~ 30) before W_head_T, so p_T
            keeps its sharp native temperature."""
        W = self.student.window
        self.head_w = (self.teacher.head.weight if self.teacher.head is not None
                       else self.teacher.embed.weight).to(self.device)
        d_S = self.student.blocks[0].psi.shape[0]
        # collect teacher last-position states for the projector
        u_T_buf, xs = [], []
        with torch.no_grad():
            for _ in range(64):
                xb = self.real_windows(data, batch)
                xs.append(xb)
                u_T_buf.append(self.teacher.features(xb, full=True)[:, -1])
        self._last_x = xs[-1]
        self._last_uT = u_T_buf[-1].float()
        u_T = torch.cat(u_T_buf)                     # (N, d_T)
        self.P = fit_task_projection(self.head_w, d_S).to(self.device)
        self.teacher_proj_norm = (u_T @ self.P.t()).norm(dim=1).mean().item()
        with torch.no_grad():
            y_T = self.teacher._logits(
                self.teacher.features(self._last_x, full=True)[:, -1])
            u_Tp = ((self._last_uT @ self.P.t()) @ self.P) @ self.head_w.t()
            acc = (u_Tp.argmax(-1) == y_T.argmax(-1)).float().mean().item()
            print(f"  stage1: task-aware P ceiling (argmax agreement with "
                  f"teacher): {acc*100:.1f}% | |u_T P^T| mean "
                  f"{self.teacher_proj_norm:.2f}", flush=True)

        # trainable: stem + attention + mix + first n_comp_blocks
        comp = [self.student.omega, self.student.theta, self.student.lift.weight]
        if self.student.pos_mode == "harmonic":
            comp += [self.student.omega_p, self.student.theta_p]
        comp += list(self.student.attn_layers[0].parameters()) \
            + list(self.student.mix_blocks[0].parameters())
        for i in range(self.n_comp_blocks):
            comp += list(self.student.blocks[i].parameters())
        for p in self.student.parameters():
            p.requires_grad_(False)
        for p in comp:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(comp, lr=lr, weight_decay=0.0)
        last = None
        for it in range(steps):
            pos = torch.randint(0, data.shape[0] - W - 1, (batch,), device=self.device)
            off = torch.arange(W, device=self.device)
            xb = data[pos.unsqueeze(1) + off.unsqueeze(0)].long()
            with torch.no_grad():
                u_T = self.teacher.features(xb, full=True)[:, -1].float()
                target = u_T @ self.P.t()            # u_T P^T (d_S)
            u_S = compressor_forward(self.student, xb, self.n_comp_blocks)
            loss = distill_loss(u_S, target, lam_norm=self.lam_norm)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = loss.item()
        for p in comp:
            p.requires_grad_(False)
        print(f"  stage1: compressor distill loss {last:.4f} "
              f"({steps} steps, P {tuple(self.P.shape)})")
        return self.student

    # ---- Stage 2 -------------------------------------------------------
    def fry(self, data, steps=6000, batch=65536, lr=5e-4, T_kd=2.0,
            mode="both", soft_steps=None, var_pres=True, ortho_every=50,
            radius="empirical", noise=0.05, tau=None, eval_fn=None,
            real_every=5, start_it=0, opt_state=None, save_hook=None,
            save_every=500):
        """Fry the decoder stack + head on millions of analytic-target
        vectors. mode: 'softvec' | 'kl' | 'both' (soft first, then KL)."""
        buf = collect_state_buffer(self.student, data,
                                   n_comp_blocks=self.n_comp_blocks,
                                   device=self.device)
        sampler = SphericalMixup(buf, radius=radius, noise=noise)
        stack_params = []
        for i in range(self.n_comp_blocks, len(self.student.blocks)):
            stack_params += list(self.student.blocks[i].parameters())
        stack_params += list(self.student.decoder.head.parameters())
        for p in stack_params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(stack_params, lr=lr, weight_decay=0.0)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        dec = self.student.decoder
        C = dec.codes()                              # (V, K), aligned by stage 0
        tau = tau or dec.tau
        soft_steps = soft_steps if soft_steps is not None \
            else (steps // 2 if mode == "both" else steps)
        blocks = [self.student.blocks[i]
                  for i in range(self.n_comp_blocks, len(self.student.blocks))]
        last = None
        for it in range(start_it, steps):
            if real_every and (it + 1) % real_every == 0:
                # real-distribution anchor: fresh compressor states + analytic
                # teacher targets, same loss path (keeps the stack honest on
                # the actual u_S scale/mix of norms)
                u_syn = self.real_states(data, min(8192, batch))
            else:
                u_syn = sampler.sample(batch, self.device)
            with torch.no_grad():
                # L_T full in bf16 (rows x V x 2B ~ 67MB at B=16k) is the ONLY
                # V-sized buffer; everything downstream is computed in row
                # chunks, so the allocator steady-state stays small and the
                # cache flush is only needed at eval boundaries.
                L_T = teacher_logits_analytic(u_syn, self.P, self.head_w,
                                              self.teacher_proj_norm)                     .to(torch.bfloat16)
            # single fused pass: stack + head + variance anchors (var_pres
            # used to run a SECOND forward - this doubles as the fix)
            h = u_syn
            loss = None
            for b in blocks:
                h_in = h
                h = b(h)
                if var_pres:                         # anti-collapse anchor
                    t = 1e-2 * ((h.var() - h_in.var()) ** 2)
                    loss = t if loss is None else loss + t
            z_hat = dec.head(h)                      # (B, K)
            if mode == "kl" or (mode == "both" and it >= soft_steps):
                # chunked reverse KL: (rows x V) buffers per chunk only;
                # logp_T is derived on the fly from the bf16 L_T
                n_rows, chunk = z_hat.shape[0], 4096
                kl_total = z_hat.new_zeros(())
                for i0 in range(0, n_rows, chunk):
                    zi = z_hat[i0:i0 + chunk]
                    ni = torch.sqrt(zi.pow(2).sum(-1, keepdim=True) + dec.tau0 ** 2)
                    ls = F.log_softmax(tau * (zi @ C.t()) / ni, -1)
                    lt = F.log_softmax(L_T[i0:i0 + chunk].float() / T_kd, -1)
                    kl_total = kl_total + (ls.exp() * (ls - lt)).sum()
                t = kl_total / n_rows
                loss = t if loss is None else loss + t
            else:
                # chunked softvec target: only (chunk x K) work per slice
                cs = 4096
                z_star = torch.empty_like(z_hat)
                for i0 in range(0, z_hat.shape[0], cs):
                    p_c = F.softmax(L_T[i0:i0 + cs].float() / T_kd, -1)
                    z_star[i0:i0 + cs] = p_c @ C
                t = 1.0 - F.cosine_similarity(z_hat, z_star, dim=1).mean()
                loss = t if loss is None else loss + t
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ortho_every and (it + 1) % ortho_every == 0:
                self._orthonormalize()
            if (it + 1) % 2000 == 0:
                torch.cuda.empty_cache()   # only at eval boundaries now
            if save_hook and (it + 1) % save_every == 0:
                save_hook(it + 1, self.student, opt)
            last = loss.item()
            if eval_fn and (it + 1) % max(1, steps // 10) == 0:
                print(f"  fry {it+1}: loss {last:.4f} | {eval_fn()}", flush=True)
                torch.cuda.empty_cache()
        print(f"  stage2: final fry loss {last:.4f} ({steps} steps)")
        if save_hook:
            save_hook(steps, self.student, opt)
        return self.student

    def _orthonormalize(self):
        """Newton-Schulz step on channel-block W: W <- 1.5W - 0.5 W W^T W."""
        for i in range(self.n_comp_blocks, len(self.student.blocks)):
            b = self.student.blocks[i]
            if hasattr(b, "weight") and b.weight.requires_grad:
                with torch.no_grad():
                    W = b.weight
                    b.weight.copy_(1.5 * W - 0.5 * W @ W.t() @ W)


# ---------------------------------------------------------------- polish

def polish(student, teacher, P, data, steps=2000, lr=3e-5, device="cuda",
           T_kd=2.0, eval_fn=None):
    """End-to-end: small lr, reverse KL (S||T) vs real teacher logits + CE.
    Works for both all-position (B, W, D) and last-position (B, D) students."""
    W = student.window
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    for it in range(steps):
        pos = torch.randint(0, data.shape[0] - W - 1, (64,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        x, y = data[idx].long(), data[idx + 1].long()
        h_S = student.features(x)                        # (B, W, D) or (B, D)
        if h_S.dim() == 3:
            rows = h_S.reshape(-1, h_S.shape[-1])
            yf = y.reshape(-1)
            with torch.no_grad():
                u_T = teacher.features(x, full=True).reshape(-1, teacher.embed.weight.shape[1]).float()
        else:
            rows, yf = h_S, y[:, -1]
            with torch.no_grad():
                u_T = teacher.features(x, full=True)[:, -1].float()
        with torch.no_grad():
            L_T = u_T @ teacher_logits_w(teacher).t()    # raw d_T -> logits
        dec = student.decoder
        C = dec.codes()
        z_S = dec.head(rows)
        n_rows, chunk = z_S.shape[0], 8192
        kl_total = z_S.new_zeros(())
        ce_total = z_S.new_zeros(())
        for i0 in range(0, n_rows, chunk):
            zi = z_S[i0:i0 + chunk]
            ni = torch.sqrt(zi.pow(2).sum(-1, keepdim=True) + dec.tau0 ** 2)
            logits_S = dec.tau * (zi @ C.t()) / ni
            ls = F.log_softmax(logits_S, -1)
            lt = F.log_softmax(L_T[i0:i0 + chunk] / T_kd, -1)
            kl_total = kl_total + (ls.exp() * (ls - lt)).sum()
            ce_total = ce_total + F.cross_entropy(
                logits_S, yf[i0:i0 + chunk], reduction="sum")
        loss = 0.5 * (kl_total / n_rows) + 0.5 * (ce_total / n_rows)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_fn and (it % max(1, steps // 10) == 0 or it == steps - 1):
            print(f"  polish {it}: loss {loss.item():.4f} | {eval_fn()}", flush=True)
            torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    return student


def teacher_logits_w(teacher):
    return teacher.head.weight if teacher.head is not None else teacher.embed.weight


# ------------------------------------------------------- end-to-end Top-K KD

def train_e2e_topk(student, teacher, data, steps=8000, batch=256, lr=3e-4,
                   K=32, T_kd=2.0, ce_weight=0.1, eval_fn=None, device="cuda",
                   save_hook=None, save_every=500, start_it=0, opt_state=None):
    """End-to-end distillation of the FULL student graph (1-7) against the
    teacher's Top-K logits - no latent split u_S ~= u_T P^T, no softvec.

    Why this beats the split protocol:
      * the stack (6) keeps its true role: untangling the raw window code
        into linearly separable phases (it costs <2% of the attention FLOPs
        when run on all window positions);
      * the teacher target is EXACT (44.5% ceiling), not the 41.7% via a
        projector and not the ~16% through the student's own compressor;
      * all-position training (train_allpos=True) gives 64 training signals
        per window - the from-scratch baselines already prove this protocol.

    loss = reverse-KL(p_S || p_T) over the teacher's top-K tokens
         + ce_weight * CE(student, teacher argmax),
    both temperatures T_kd (student logits carry the head's tau=10 scale and
    are divided by T_kd as well, so both sides share one temperature).
    """
    W = student.window
    student.train()
    params = [p for p in student.parameters() if p.requires_grad or True]
    for p in params:
        p.requires_grad_(True)
    codebook_thaw = int(0.25 * steps)                # protect the gram-aligned
    cb = student.decoder.code_cache                  # codebook from early
    cb.requires_grad_(False)                         # gradient shock
    main_params = [p for n, p in student.named_parameters()
                   if n != "decoder.code_cache"]
    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": lr},
         {"params": [cb], "lr": lr * 0.1}],          # codebook: gentle lr
        betas=(0.8, 0.999), weight_decay=0.0)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    dec = student.decoder
    C = dec.codes()                                  # (V, Kd) gram-aligned
    tau, tau0 = dec.tau, dec.tau0
    def windows(count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        return data[pos.unsqueeze(1) + off.unsqueeze(0)].long()
    last = None
    for it in range(start_it, steps):
        x = windows(batch)
        with torch.no_grad():
            with torch.amp.autocast("cuda"):         # teacher is cheap in bf16
                h_T = teacher.features(x, full=True)  # (B, W, d_T)
            logits_T = teacher._logits(h_T)           # (B, W, V) bf16, exact
            y_T = logits_T.argmax(-1)                 # teacher argmax labels
        h_S = student.features(x)                     # (B, W, D) ALL positions
        rows = h_S.reshape(-1, h_S.shape[-1])
        z = dec.head(rows)                            # (rows, Kd)
        # FULL-vocab chunked reverse KL + CE (no Top-32 tail hole): the
        # teacher's full logits are already materialized in bf16, so the
        # exact 2048-way distributions cost nothing extra. Student logits
        # keep the head's NATIVE scale (tau * cos / norm, range ~[-10,10]);
        # BOTH sides are divided by T_kd; the KL term carries T_kd^2 so the
        # temperature division does not shrink the gradients 4x.
        logits_T_flat = logits_T.reshape(-1, logits_T.shape[-1])   # bf16
        y_flat = y_T.reshape(-1)
        n_rows, chunk = z.shape[0], 4096
        kl_total, ce_total = z.new_zeros(()), z.new_zeros(())
        for i0 in range(0, n_rows, chunk):
            zi = z[i0:i0 + chunk]
            ni = torch.sqrt(zi.pow(2).sum(-1, keepdim=True) + tau0 ** 2)
            raw_S = tau * (zi @ C.t()) / ni                        # native scale
            logp_S = F.log_softmax(raw_S / T_kd, -1)
            logp_T = F.log_softmax(logits_T_flat[i0:i0 + chunk].float() / T_kd, -1)
            kl_total = kl_total + (logp_S.exp() * (logp_S - logp_T)).sum()
            ce_total = ce_total - logp_S.gather(
                -1, y_flat[i0:i0 + chunk].unsqueeze(-1)).sum()
        kl = kl_total / n_rows
        ce = ce_total / n_rows
        loss = (T_kd ** 2) * kl + ce_weight * T_kd * ce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
        if it == codebook_thaw:
            cb.requires_grad_(True)                  # thaw at 25%, lr*0.1
        if save_hook and (it + 1) % save_every == 0:
            save_hook(it + 1, student, opt)
        if eval_fn and (it + 1) % max(1, steps // 10) == 0:
            torch.cuda.empty_cache()
            print(f"  e2e {it+1}: loss {last:.4f} | {eval_fn()}", flush=True)
    if save_hook:
        save_hook(steps, student, opt)
    print(f"  e2e: final loss {last:.4f} ({steps} steps)", flush=True)
    return student


class GraphedKDStep:
    """CUDA-graph capture of the WHOLE e2e KD step: teacher bf16 forward +
    student forward + full-vocab reverse-KL/CE + fused capturable Adam.

    Rationale: the eager e2e step runs hundreds of small kernels (123.5 ms
    at B=256 on the 3050, GPU mostly idle at launch gaps -> 70C). The
    champion trainer replays one graph (~21-58 ms, 86C saturated). All
    shapes here are static (fixed B, fixed chunk loops), so the whole step
    captures cleanly. Loss must be read via .item() OUTSIDE the graph.
    """

    def __init__(self, student, teacher, opt, W=64, B=256, K_d=32, T_kd=2.0,
                 ce_weight=0.1, warmup=3):
        self.student, self.teacher, self.opt = student, teacher, opt
        self.B, self.W, self.T_kd, self.ce_weight = B, W, T_kd, ce_weight
        dec = student.decoder
        self.C = dec.codes().detach()
        self.tau, self.tau0 = dec.tau, dec.tau0
        dev = next(student.parameters()).device
        self.x = torch.zeros(B, W, dtype=torch.long, device=dev)
        self.y_argmax = torch.zeros(B * W, dtype=torch.long, device=dev)
        self.logits_T = torch.zeros(B * W, self.C.shape[0],
                                    dtype=torch.bfloat16, device=dev)
        # ---- warmup (build lazy kernels/autocast caches outside capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._loss_fn(self.x, self.y_argmax, self.logits_T, backward=True)
        torch.cuda.current_stream().wait_stream(s)
        # ---- capture
        g = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(g):
            self.static_loss = self._loss_fn(self.x, self.y_argmax,
                                             self.logits_T, backward=True)
        self.graph = g

    def _loss_fn(self, x, y_argmax, logits_T_flat, backward):
        h_T = self.teacher.features(x, full=True)
        with torch.no_grad():
            lt = self.teacher._logits(h_T)                     # (B, W, V) bf16
            logits_T_flat.copy_(lt.reshape(-1, lt.shape[-1]))
            y_argmax.copy_(lt.argmax(-1).reshape(-1))
        h_S = self.student.features(x)
        rows = h_S.reshape(-1, h_S.shape[-1])
        z = self.student.decoder.head(rows)
        C, tau, tau0, T_kd = self.C, self.tau, self.tau0, self.T_kd
        # fused reverse-KL + CE: ONE log_softmax per side. The CE term is
        # taken at the KD temperature (CE_T = CE_raw / T_kd: same argmax,
        # gradients scaled by 1/T_kd) and re-scaled by ce_weight*T_kd, so
        # the effective CE gradient equals the original spec exactly.
        n_rows, chunk = z.shape[0], 4096
        kl_total, ce_total = z.new_zeros(()), z.new_zeros(())
        for i0 in range(0, n_rows, chunk):
            zi = z[i0:i0 + chunk]
            ni = torch.sqrt(zi.pow(2).sum(-1, keepdim=True) + tau0 ** 2)
            raw_S = tau * (zi @ C.t()) / ni
            logp_S = F.log_softmax(raw_S / T_kd, -1)
            logp_T = F.log_softmax(logits_T_flat[i0:i0 + chunk].float() / T_kd, -1)
            kl_total = kl_total + (logp_S.exp() * (logp_S - logp_T)).sum()
            ce_total = ce_total - logp_S.gather(
                -1, y_argmax[i0:i0 + chunk].unsqueeze(-1)).sum()
        loss = (T_kd ** 2) * kl_total / n_rows             + self.ce_weight * T_kd * ce_total / n_rows
        if backward:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return loss

    def step(self, x, teacher_labels_src=None):
        """Copy real window batch into static buffers, replay, return loss."""
        self.x.copy_(x)
        self.graph.replay()
        return self.static_loss


def bucket_kl_ce(raw_S, logits_T_chunk, y_chunk, T_kd=2.0, K=32):
    """Exact 2048-way reverse KL via the 33-outcome bucket (top-K + rest).

    raw_S: (c, V) NATIVE student logits, already tau*cos/norm-normalized.
    Reverse KL needs the FULL student distribution (else the student could
    dump mass into the 2016-token tail) and log p_T of the tail would be
    -inf if zeroed. The bucket closes both holes exactly:
      p_S = [p_S(top-K tokens), 1 - sum]   (33 outcomes, sums to 1)
      p_T = [p_T(top-K tokens), 1 - sum]
    NOTE: the KL part is an APPROXIMATION of the exact reverse KL: it is
    exact for the top-K tokens, and the bucket cross-term enforces the
    TOTAL tail mass (the mass-dumping pathology), but the student's shape
    WITHIN the 2016-token tail is not penalized. The CE part is exact.
    For the exact KL use the full-vocab chunked path (default).
    Returns (kl_row_sum, ce_row_sum) over the chunk.
    """
    lS = raw_S                                               # native scale
    # temperature-scaled copies for the KL
    lS_T = lS / T_kd
    lT_T = logits_T_chunk.float() / T_kd
    topv, topi = lT_T.topk(K, dim=-1)                        # teacher top-K
    logZ_S = torch.logsumexp(lS_T, -1, keepdim=True)
    logZ_T = torch.logsumexp(lT_T, -1, keepdim=True)
    aS = lS_T.gather(-1, topi) - logZ_S                      # log p_S top-K
    aT = topv - logZ_T                                       # log p_T top-K
    mask = torch.zeros_like(lS_T, dtype=torch.bool).scatter_(-1, topi, True)
    log_rest_S = torch.logsumexp(lS_T.masked_fill(mask, -torch.inf), -1,
                                 keepdim=True) - logZ_S
    log_rest_T = torch.logsumexp(lT_T.masked_fill(mask, -torch.inf), -1,
                                 keepdim=True) - logZ_T
    logp_S = torch.cat([aS, log_rest_S], -1)                 # (c, K+1), sums 1
    logp_T = torch.cat([aT, log_rest_T], -1)
    p_S = logp_S.exp()
    kl = (p_S * (logp_S - logp_T)).sum(-1)                   # (c,)
    # exact CE in native student scale: label y is the teacher argmax,
    # which is topi[:, 0] by construction
    logZ_raw = torch.logsumexp(lS, -1, keepdim=True)
    ce = -(lS.gather(-1, y_chunk.unsqueeze(-1)) - logZ_raw).squeeze(-1)
    return kl.sum(), ce.sum()


def train_e2e_v3(student, teacher, data, steps=24000, batch=256, lr=3e-4,
                 T_kd=4.0, kd_weight=1.0, kd_every=2, kd_until_frac=0.7,
                 eval_fn=None, device="cuda", save_hook=None, save_every=500,
                 start_it=0, opt_state=None):
    """v3 protocol (DKD + Stochastic/Curriculum KD):

      loss = CE(logits_S, y_true)                                  every step
           + kd_weight * kd_every * T_kd^2
             * ForwardKL_non_target(logits_S/T, logits_T/T)        on KD steps

    * CE is taken ONLY at the true label - the teacher's 44.5% wrong argmax
      never fights the ground truth (the v2 mistake).
    * The KL term is Forward (mean-seeking, transfers the soft structure of
      the alternatives) and NON-TARGET: the y_true column is masked to -inf
      in BOTH distributions before the softmax, so what transfers is the
      pure dark knowledge "given not-y, how do alternatives rank".
    * Stochastic KD: the teacher runs every kd_every-th step (lambda
      pre-multiplied by kd_every so the average gradient scale is kept) and
      switches OFF entirely after kd_until_frac of training (curriculum:
      fast pure-CE finish at champion speed).
    * T^2 factor keeps the KD gradient norm flat during any T annealing.
    """
    W = student.window
    student.train()
    for p in student.parameters():
        p.requires_grad_(True)
    codebook_thaw = int(0.25 * steps)
    cb = student.decoder.code_cache
    cb.requires_grad_(False)
    main_params = [p for n, p in student.named_parameters()
                   if n != "decoder.code_cache"]
    opt = torch.optim.AdamW(
        [{"params": main_params, "lr": lr},
         {"params": [cb], "lr": lr * 0.1}],
        betas=(0.8, 0.999), weight_decay=0.0)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    dec = student.decoder
    C = dec.codes()
    tau, tau0 = dec.tau, dec.tau0
    def windows(count):
        pos = torch.randint(0, data.shape[0] - W - 1, (count,), device=device)
        off = torch.arange(W, device=device)
        idx = pos.unsqueeze(1) + off.unsqueeze(0)
        return data[idx].long(), data[idx + 1].long()
    last = None
    for it in range(start_it, steps):
        if it == codebook_thaw:
            cb.requires_grad_(True)
        x, y = windows(batch)
        kd_on = (it % kd_every == 0) and it < int(kd_until_frac * steps)
        with torch.no_grad():
            if kd_on:
                with torch.amp.autocast("cuda"):
                    logits_T = teacher._logits(teacher.features(x, full=True))
        h_S = student.features(x)                        # (B, W, D) allpos
        rows = h_S.reshape(-1, h_S.shape[-1])
        y_flat = y.reshape(-1)
        n_rows, chunk = rows.shape[0], 8192
        ce_total = rows.new_zeros(())
        z_rows = []
        for i0 in range(0, n_rows, chunk):
            z = dec.head(rows[i0:i0 + chunk])
            z_rows.append(z)
            ni = torch.sqrt(z.pow(2).sum(-1, keepdim=True) + tau0 ** 2)
            raw_S = tau * (z @ C.t()) / ni
            ce_total = ce_total + F.cross_entropy(raw_S, y_flat[i0:i0 + chunk],
                                                  reduction="sum")
        ce = ce_total / n_rows
        loss = ce
        if kd_on:
            lt_flat = logits_T.reshape(-1, logits_T.shape[-1]).float()
            kl_total = rows.new_zeros(())
            for i0 in range(0, n_rows, chunk):
                z = z_rows[i0 // chunk]
                ni = torch.sqrt(z.pow(2).sum(-1, keepdim=True) + tau0 ** 2)
                raw_S = tau * (z @ C.t()) / ni
                ycol = y_flat[i0:i0 + chunk].unsqueeze(-1)
                # mask the true-label column in BOTH raw distributions,
                # then one log_softmax each: pure non-target distributions
                lS_m = (raw_S / T_kd).scatter(-1, ycol, -torch.inf)
                lT_m = (lt_flat[i0:i0 + chunk] / T_kd).scatter(-1, ycol,
                                                               -torch.inf)
                logp_S = F.log_softmax(lS_m, -1)
                logp_T = F.log_softmax(lT_m, -1)
                # forward KL (teacher-weighted, mean-seeking). nan_to_num:
                # the masked y column is -inf in BOTH log-probs, so the raw
                # difference is -inf-(-inf)=NaN there (exp(logp_T)=0 -> the
                # term must be 0); also neutralizes degenerate teacher rows
                # whose non-target logits are all -inf (no dark knowledge).
                diff = logp_T.exp() * (logp_T - logp_S)
                kl_total = kl_total + torch.nan_to_num(diff, nan=0.0).sum()
            kl = kl_total / n_rows
            loss = loss + kd_weight * kd_every * (T_kd ** 2) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
        if save_hook and (it + 1) % save_every == 0:
            save_hook(it + 1, student, opt)
        if eval_fn and (it + 1) % max(1, steps // 10) == 0:
            torch.cuda.empty_cache()
            print(f"  v3 {it+1}: loss {last:.4f} | {eval_fn()}", flush=True)
    if save_hook:
        save_hook(steps, student, opt)
    print(f"  v3: final loss {last:.4f} ({steps} steps)", flush=True)
    return student
