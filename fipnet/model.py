"""FIP-Net modules: harmonic encoder, isometric blocks, low-rank phase
bottleneck head, and an MNIST classifier."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import fip_wave, fip_tri, GAMMA, BETA, PERIOD
import os

BIAS_PERIOD = PERIOD  # biases initialized U[-2, 2] -> uniform phase on R/4Z


def log_spaced_frequencies(dim, omega_min, omega_max):
    """omega_i = omega_min * (omega_max/omega_min)^(i/(dim-1))."""
    i = torch.arange(dim, dtype=torch.float64)
    omega = omega_min * (omega_max / omega_min) ** (i / (dim - 1))
    return omega.float()


def _uniform_phase_init(tensor):
    """theta ~ U[-2, 2]: guarantees uniform phase at init for any input."""
    with torch.no_grad():
        tensor.uniform_(-BIAS_PERIOD / 2.0, BIAS_PERIOD / 2.0)
    return tensor


class HarmonicEncoder(nn.Module):
    """Discrete token/value s -> hidden vector without an embedding table.

    h0 = phi(s * omega + theta); omega fixed log-spaced, theta ~ U[-2, 2].
    Parameter cost O(D) instead of O(V * D).
    """

    def __init__(self, dim, omega_min=0.4, omega_max=9.7,
                 gamma=GAMMA, beta=BETA):
        super().__init__()
        self.dim = dim
        self.gamma = gamma
        self.beta = beta
        self.register_buffer("omega", log_spaced_frequencies(dim, omega_min, omega_max))
        self.theta = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.theta)

    def forward(self, values):
        z = values.to(torch.float32).unsqueeze(-1) * self.omega + self.theta
        return fip_wave(z, self.gamma, self.beta)

    @torch.no_grad()
    def value_histogram(self, values, vocab_size=256):
        """counts[b, s] = (# pixels equal to s in sample b) / N.

        Integer pixel values only; pooled harmonic encoding is exactly
        counts @ LUT, so the (B, N, D) intermediate is never materialized.
        """
        values = values.long()
        B, N = values.shape
        idx = values + torch.arange(B, device=values.device,
                                    dtype=values.dtype).unsqueeze(1) * vocab_size
        counts = torch.bincount(idx.reshape(-1),
                                minlength=B * vocab_size).to(torch.float32)
        return (counts / N).view(B, vocab_size)

    def pooled(self, values, vocab_size=256):
        """Mean-pooled harmonic encoding of integer values (B, N) -> (B, D).

        LUT[v, i] = phi(v * omega_i + theta_i) is differentiable w.r.t.
        theta; gradients flow through the matmul, not the histogram.
        """
        counts = self.value_histogram(values, vocab_size)
        v = torch.arange(vocab_size, dtype=torch.float32, device=self.omega.device)
        lut = fip_wave(v.unsqueeze(-1) * self.omega + self.theta,
                       self.gamma, self.beta)                       # (V, D)
        return counts @ lut


class IsometricBlock(nn.Module):
    """h' = phi(W h + b), W strictly orthogonal (gain 1) at init,
    b ~ U[-2, 2] (uniform phase). No normalization, no residual.
    activation: "parabolic" (two-parametric) or "triangular" (phi' = +-1,
    block Jacobian exactly orthogonal).
    """

    def __init__(self, dim, gamma=GAMMA, beta=BETA, activation="parabolic"):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.gamma = gamma
        self.beta = beta
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight, gain=1.0)
        _uniform_phase_init(self.bias)

    def forward(self, h):
        z = F.linear(h, self.weight, self.bias)
        if self.activation == "parabolic":
            return fip_wave(z, self.gamma, self.beta)
        if self.activation == "triangular":
            return fip_tri(z)
        raise ValueError(self.activation)


class LowRankPhaseBottleneck(nn.Module):
    """Low-rank phase projection head (formerly "interferometric decoder").

    z_hat = W_head h with W_head in R^{K x D} (O(KD) parameters, K << V),
    trained with phase-matching cosine loss against fixed per-class
    frequency codes. Inference decodes by resonance (argmax over cached
    codes, O(VK)) or, in "binary" codebook mode, by phase unwrapping
    (O(K): digit-by-digit from low to high binary frequency).
    """

    def __init__(self, dim, vocab, code_dim=16, mode="binary",
                 omega_min=0.4, omega_max=9.7, gamma=GAMMA, beta=BETA,
                 tau=10.0, tau0=0.1, trainable_codes=False):
        super().__init__()
        self.vocab = vocab
        self.code_dim = code_dim
        self.mode = mode
        self.gamma = gamma
        self.beta = beta
        self.tau = tau      # scaled-cosine CE temperature
        self.tau0 = tau0    # smoothed-norm floor, removes the 1/|z_hat| singularity
        if mode == "binary":
            # omega_k = 2^k * step, step = (4/beta) / 2^bits.  The parabolic
            # wave has period 4/beta in its argument, so feature k depends
            # only on v mod 2^(bits-k) -> enables O(K) phase unwrapping.
            bits = max(1, math.ceil(math.log2(max(vocab, 2))))
            self.bits = bits
            self.codebook_size = 2 ** bits
            step = PERIOD / (self.beta * self.codebook_size)
            omega = step * (2.0 ** torch.arange(code_dim, dtype=torch.float32))
            self.register_buffer("omega_out", omega)
        elif mode == "log":
            self.register_buffer(
                "omega_out", log_spaced_frequencies(code_dim, omega_min, omega_max))
        elif mode == "quad":
            # quadrature (sin/cos-like) pairs: c_{v,2j} = phi(v*w_j),
            # c_{v,2j+1} = phi(v*w_j + T/4). Even K required; the T/4 phase
            # offset keeps code norms nearly constant and classes separated.
            n_pairs = code_dim // 2
            self.n_pairs = n_pairs
            omega = (PERIOD / vocab) * (2.0 ** torch.arange(n_pairs, dtype=torch.float32))
            self.register_buffer("omega_out", omega)
        elif mode == "fourier":
            # zero-mean random-Fourier codes for large vocabularies: sinusoidal
            # (quad/vernier) structure cannot keep V >> K^2 classes separated
            # (Welch bound), so start from seeded log-spaced frequencies and
            # refine the codebook numerically on first use (see codes()).
            g = torch.Generator().manual_seed(0)
            omega = log_spaced_frequencies(code_dim, 2.0 / vocab, vocab / 4.0)
            omega = omega * (1.0 + 0.15 * (torch.rand(code_dim, generator=g) * 2 - 1))
            self.register_buffer("omega_out", omega)
            self.register_buffer("theta_out", torch.rand(code_dim, generator=g) * PERIOD)
        else:
            raise ValueError(mode)
        self.head = nn.Linear(dim, code_dim, bias=False)
        self.trainable_codes = trainable_codes
        self.register_buffer("code_cache", torch.zeros(vocab, code_dim))
        self._codes_valid = False

    def _code_phase(self, v):
        """Phase arguments for class indices v (tensor or int); (..., K)."""
        v = torch.as_tensor(v, dtype=torch.float32, device=self.omega_out.device)
        off = 0.5 if self.mode == "binary" else 1.0   # +off: avoid zero codes
        return (v.unsqueeze(-1) + off) * self.omega_out

    def _raw_code_matrix(self, size):
        """Unnormalized code rows for indices 0..size-1, shape (size, K)."""
        v = torch.arange(size, dtype=torch.float32, device=self.omega_out.device)
        if self.mode == "quad":
            base = (v.unsqueeze(-1) + 0.5) * self.omega_out          # (size, n_pairs)
            rows = [fip_wave(base, self.gamma, self.beta, theta=off)
                    for off in (0.0, 1.0)]                            # T/4 pair
            return torch.cat(rows, dim=-1)
        if self.mode == "fourier":
            c = fip_wave(v.unsqueeze(-1) * self.omega_out + self.theta_out,
                         self.gamma, self.beta)
            return c - c.mean(0, keepdim=True)               # zero-mean features
        return fip_wave(self._code_phase(v), self.gamma, self.beta)

    def optimize_codebook(self, steps=600, lr=0.03, margin=0.45, cache_path=None):
        """One-time numerical refinement of the fourier codebook: minimize
        squared off-diagonal cosines with a hinge on |cos| > margin and a
        row-norm regularizer. Deterministic (seeded init). For V >> K^2 the
        raw sinusoidal construction has near-parallel code pairs (cos ~ 0.99
        at V=2048, K=16); refinement brings max cosine to ~0.5 (K=32).
        For large V the off-diagonal term is estimated on a random row
        subsample each step (exact for V <= 8192)."""
        if cache_path and os.path.exists(cache_path):
            self.code_cache.copy_(torch.load(cache_path, map_location=self.code_cache.device))
            self._codes_valid = True
            return
        if self.mode != "fourier":
            raise ValueError("codebook optimization requires mode='fourier'")
        c = self._raw_code_matrix(self.vocab).clone().requires_grad_(True)
        opt = torch.optim.Adam([c], lr=lr)
        g = torch.Generator(device="cpu").manual_seed(7)
        full = self.vocab <= 8192
        if full:
            mask = ~torch.eye(self.vocab, dtype=torch.bool, device=c.device)
        n_sub = min(4096, self.vocab)
        # enable_grad: codes() may be called lazily inside a caller's
        # no_grad context (e.g. first predict()) - the refinement needs
        # its own autograd regardless of the ambient context.
        with torch.enable_grad():
            for _ in range(steps):
                n = c.norm(dim=-1, keepdim=True)
                cn = c / n.clamp_min(0.5)
                if full:
                    off = (cn @ cn.t())[mask]
                else:
                    idx = torch.randperm(self.vocab, generator=g)[:n_sub].to(c.device)
                    sub = cn[idx]
                    gr = sub @ sub.t()
                    off = gr[~torch.eye(n_sub, dtype=torch.bool, device=c.device)]
                loss = ((off ** 2).mean()
                        + 10.0 * F.relu(off.abs() - margin).pow(2).mean()
                        + ((n - 1.0) ** 2).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        with torch.no_grad():
            c = c.detach()
            c = c / c.norm(dim=-1, keepdim=True)   # exact unit rows
        self.code_cache.copy_(c)
        self._codes_valid = True
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.code_cache.detach().cpu(), cache_path)

    def codes(self, cache_path=None):
        """Unit-norm V x K code matrix, cached.

        Normalizing the dictionary equalizes class scales and makes the
        dot-product argmax identical to the cosine argmax (consistent
        training/decoding for every loss mode).
        """
        if not self._codes_valid:
            if self.mode == "fourier":
                self.optimize_codebook(cache_path=cache_path)
                if self.trainable_codes:
                    # promote the refined codebook to a learnable parameter:
                    # O(V*K) parameters (K << V), still 4-8x fewer than a
                    # V x D output matrix; resonance argmax decoding unchanged
                    data = self.code_cache.data
                    del self._buffers["code_cache"]
                    self.code_cache = nn.Parameter(data)
                self._codes_valid = True
                return self.code_cache
            c = self._raw_code_matrix(self.vocab)
            self.code_cache.copy_(c / c.norm(dim=-1, keepdim=True))
            self._codes_valid = True
        return self.code_cache

    def project(self, h):
        return self.head(h)

    def loss(self, h, targets, mode="ce", eps=1e-6):
        """Training loss in the phase code space (codes are unit-norm).

        mode="ce": scaled cosine cross-entropy (CosFace/CLIP-style with a
            fixed codebook): s_v = tau * <z_hat, c_v> / sqrt(|z_hat|^2 + tau0^2).
            Bounded logits and the tau0 floor remove the 1/|z_hat|
            singularity; CE adds repulsion from rival classes (default).
        mode="mse": 0.5*||z_hat - c_y||^2 — linear gradient, smooth at
            zero, no explosion possible; argmax decoding is exactly
            equivalent while ||c_v|| = 1.
        mode="cos": spec phase-matching loss 1 - cosine(z_hat, z(y));
            singular as |z_hat| -> 0 (kept for reference).
        """
        z_hat = self.head(h)
        if mode == "ce":
            norm = torch.sqrt(z_hat.pow(2).sum(-1, keepdim=True) + self.tau0 ** 2)
            n = z_hat.shape[0]
            if n <= 8192:
                logits = self.tau * (z_hat @ self.codes().t()) / norm
                return F.cross_entropy(logits, targets)
            # chunked scaled-cosine CE: the (chunk x V) logits matrix is
            # materialized and back-propagated block by block, so the
            # transient CE buffers never exceed one chunk (VRAM-flat for
            # all-position training and large vocabularies)
            codes_t = self.codes().t()
            chunk = 8192
            total = z_hat.new_zeros(())
            for i in range(0, n, chunk):
                zi = z_hat[i:i+chunk]
                ni = norm[i:i+chunk]
                logits = self.tau * (zi @ codes_t) / ni
                total = total + F.cross_entropy(logits, targets[i:i+chunk],
                                                reduction="sum")
            return total / n
        if mode == "mse":
            return F.mse_loss(z_hat, self.codes()[targets])
        z = self.codes()[targets]
        cos = (z_hat * z).sum(-1) / (z_hat.norm(dim=-1) * z.norm(dim=-1) + eps)
        return (1.0 - cos).mean()

    def decode(self, h):
        """Resonance decoding: argmax_v <z_hat, c_v> over unit-norm codes,
        O(V*K). Consistent with every loss mode (cos = dot for unit codes).
        Chunked over rows: the (rows x V) score matrix is materialized in
        blocks of 4096 rows, so all-position eval (B*W rows) and large
        vocabularies never spike VRAM."""
        z_hat = self.head(h)
        codes_t = self.codes().t()
        rows = z_hat.shape[:-1]
        flat = z_hat.reshape(-1, z_hat.shape[-1])
        out = torch.empty(flat.shape[0], dtype=torch.long, device=flat.device)
        chunk = 4096
        for i in range(0, flat.shape[0], chunk):
            out[i:i+chunk] = torch.argmax(flat[i:i+chunk] @ codes_t, dim=-1)
        return out.reshape(rows)

    def decode_unwrap(self, h):
        """Phase unwrapping decoding, O(K), from hidden states."""
        return self.unwrap_from_z(self.head(h))

    def unwrap_from_z(self, z_hat):
        """Phase unwrapping decoder, O(K): digit-by-digit from the finest
        binary frequency (LSB) to the coarsest. Requires mode="binary" and
        a power-of-two vocab (else result is clamped into [0, vocab))."""
        if self.mode != "binary":
            raise ValueError("unwrap decoding requires mode='binary'")
        table = self._raw_code_matrix(self.codebook_size)
        table = table / table.norm(dim=-1, keepdim=True)
        r = torch.zeros(z_hat.shape[:-1], dtype=torch.long, device=z_hat.device)
        for k in range(self.bits - 1, -1, -1):
            block = 2 ** (self.bits - 1 - k)           # weight of the unknown bit
            cand = torch.stack([r, r + block], dim=-1)  # (..., 2) candidate v mod 2^(bits-k)
            cand_z = table[cand, k]                     # (..., 2)
            dist = (z_hat[..., k].unsqueeze(-1) - cand_z).abs()
            bit = dist.argmin(dim=-1)
            r = r + bit * block
        return r.clamp(max=self.vocab - 1)


# Backwards-compatible alias
InterferometricDecoder = LowRankPhaseBottleneck


class FIPNet(nn.Module):
    """Full FIP-Net: harmonic encoder -> L isometric blocks -> phase head."""

    def __init__(self, vocab, dim=64, depth=6, code_dim=16,
                 gamma=GAMMA, beta=BETA, omega_min=0.4, omega_max=9.7,
                 activation="parabolic", decoder_mode="binary"):
        super().__init__()
        self.vocab = vocab
        self.encoder = HarmonicEncoder(dim, omega_min, omega_max, gamma, beta)
        self.blocks = nn.ModuleList(
            IsometricBlock(dim, gamma, beta, activation) for _ in range(depth))
        self.decoder = LowRankPhaseBottleneck(dim, vocab, code_dim, decoder_mode,
                                              omega_min, omega_max, gamma, beta)
        self.gamma = gamma

    def features(self, tokens):
        h = self.encoder(tokens)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, tokens, targets=None):
        if targets is None:
            targets = tokens
        return self.decoder.loss(self.features(tokens), targets)

    def predict(self, tokens):
        return self.decoder.decode(self.features(tokens))


class TokenMixer(nn.Module):
    """Learned linear mixing across T spatial tokens: h'_t = sum_s A[t,s] h_s.

    Minimal deterministic stand-in for attention (the architecture provides
    no context mixing on its own); a full attention layer can replace it.
    """

    def __init__(self, n_tokens):
        super().__init__()
        self.mix = nn.Parameter(torch.empty(n_tokens, n_tokens))
        nn.init.orthogonal_(self.mix)

    def forward(self, tokens):
        # tokens: (B, T, D)
        return torch.einsum("ts,btd->bsd", self.mix, tokens)


class FIPMnist(nn.Module):
    """FIP-Net classifier for MNIST.

    input_mode="harmonic": spec-pure per-pixel phi(s*omega+theta),
        mean-pooled over all pixels via histogram x LUT. Order-free
        (bag-of-pixel-values), spatial structure is lost.
    input_mode="harmonic-spatial": per-cell (7x7 grid of 4x4 pixels)
        histogram x LUT -> token map (B, 49, D) -> learned token mixing
        -> mean-pool. Restores spatial information with O(1) parameters.
    input_mode="linear": orthogonal-initialized linear projection 784 -> D
        on normalized [0,1] pixels, followed by phi.

    loss_mode: "ce" (cross-entropy over resonance logits, default) or
    "cos" (spec phase-matching cosine loss).
    """

    CELL = 4   # 28/4 = 7 -> 49 cells

    def __init__(self, n_classes=10, dim=128, depth=6, code_dim=8,
                 gamma=GAMMA, beta=BETA, input_mode="harmonic-spatial",
                 activation="parabolic", loss_mode="ce",
                 omega_min=1.0/255, omega_max=12.0/255, tau=10.0, tau0=0.1):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.input_mode = input_mode
        self.activation = activation
        self.loss_mode = loss_mode
        self.gamma = gamma
        self.beta = beta
        if input_mode in ("harmonic", "harmonic-spatial"):
            # pixel-domain LUT frequencies: max phase = beta*255*omega_max ~ 1.9
            # periods over the whole [0,255] range — no phase aliasing
            self.encoder = HarmonicEncoder(dim, omega_min, omega_max, gamma, beta)
        elif input_mode == "linear":
            self.input_proj = nn.Linear(28 * 28, dim, bias=False)
            nn.init.orthogonal_(self.input_proj.weight)   # unit gain
            self.input_bias = nn.Parameter(torch.zeros(dim))
            _uniform_phase_init(self.input_bias)          # uniform phase
        else:
            raise ValueError(input_mode)
        if input_mode == "harmonic-spatial":
            n_cells = (28 // self.CELL) ** 2
            self.mixer = TokenMixer(n_cells)
        self.blocks = nn.ModuleList(
            IsometricBlock(dim, gamma, beta, activation) for _ in range(depth))
        self.decoder = LowRankPhaseBottleneck(dim, n_classes, code_dim, "quad",
                                              omega_min, omega_max, gamma, beta,
                                              tau=tau, tau0=tau0)

    def _phi(self, z):
        return (fip_wave(z, self.gamma, self.beta) if self.activation == "parabolic"
                else fip_tri(z))

    def _cell_histograms(self, values, B):
        """(B, 784) integer pixels -> (B, T, 256) per-cell histograms."""
        c = self.CELL
        g = 28 // c
        values = values.view(B, g, c, g, c)             # (B, g, c, g, c)
        counts = torch.zeros(B, g, g, 256, device=values.device)
        v = values.permute(0, 1, 3, 2, 4).reshape(B, g * g, c * c)
        offsets = torch.arange(g * g, device=values.device, dtype=torch.long) * 256
        flat_idx = (v + offsets[None, :, None]).reshape(-1)
        counts = torch.bincount(flat_idx, minlength=B * g * g * 256).to(torch.float32)
        return counts.view(B, g * g, 256) / (c * c)

    def features(self, images):
        B = images.shape[0]
        if self.input_mode == "harmonic":
            flat = images.flatten(1) * 255.0
            values = flat.round().long()
            h = self.encoder.pooled(values, vocab_size=256)      # bag-of-pixels
        elif self.input_mode == "harmonic-spatial":
            flat = images.flatten(1) * 255.0
            values = flat.round().long()
            counts = self._cell_histograms(values, B)            # (B, 49, 256)
            v = torch.arange(256, dtype=torch.float32, device=images.device)
            lut = fip_wave(v.unsqueeze(-1) * self.encoder.omega + self.encoder.theta,
                           self.gamma, self.beta)                # (256, D)
            tokens = counts @ lut                                # (B, 49, D)
            tokens = self.mixer(tokens)
            h = tokens.mean(dim=1)
        else:
            # MNIST standardization: zero mean, unit variance -> pre-activations
            # of the orthogonal-initialized projection stay within ~one period
            x = (images.flatten(1) - 0.1307) / 0.3081
            z = F.linear(x, self.input_proj.weight, self.input_bias)
            h = self._phi(z)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, images, targets):
        return self.decoder.loss(self.features(images), targets,
                                 mode=self.loss_mode)

    def predict(self, images):
        return self.decoder.decode(self.features(images))


class FIPCharLM(nn.Module):
    """Char-level FIP language model (Bengio-style window LM, no tables).

    Window of N previous chars -> structured dyadic harmonic code (no
    embedding table): omega_j = 2^j / V with quadrature T/4 pairs, lifted
    through one shared Linear(2*n_oct -> D) and cached as a V x D LUT
    gathered per token (bit-identical to per-char computation, O(1) per
    token instead of O(octaves)). Position info is preserved by the
    flatten projection N*D -> D (NO mean-pool). Then isometric blocks and
    the low-rank phase bottleneck head with K = 16 unit-norm quad codes;
    decoding by resonance argmax over codes — no V x D output matrix.
    """

    def __init__(self, vocab, window=32, dim=128, depth=6, code_dim=16,
                 n_oct=10, gamma=GAMMA, beta=BETA, loss_mode="ce",
                 tau=10.0, tau0=0.1, activation="parabolic",
                 decoder_mode="quad", trainable_codes=False):
        super().__init__()
        self.vocab = vocab
        self.window = window
        self.loss_mode = loss_mode
        self.n_oct = n_oct
        omega = (2.0 ** torch.arange(n_oct, dtype=torch.float32)) / vocab
        self.register_buffer("omega", omega)
        self.lift = nn.Linear(2 * n_oct, dim, bias=False)   # shared, O(2*n_oct*D)
        nn.init.orthogonal_(self.lift.weight)
        self.lift_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.lift_bias)
        self.flat_proj = nn.Linear(window * dim, dim, bias=False)
        nn.init.orthogonal_(self.flat_proj.weight)
        self.flat_bias = nn.Parameter(torch.zeros(dim))
        _uniform_phase_init(self.flat_bias)
        self.blocks = nn.ModuleList(
            IsometricBlock(dim, gamma, beta, activation) for _ in range(depth))
        self.decoder = LowRankPhaseBottleneck(dim, vocab, code_dim, decoder_mode,
                                              tau=tau, tau0=tau0,
                                              trainable_codes=trainable_codes)

    def _phi(self, z):
        return fip_wave(z, GAMMA, BETA)

    def char_lut(self):
        """V x D table phi-code(s) lifted to D, rebuilt each step (theta and
        lift are learnable); gradients flow through the LUT matmul."""
        z = torch.arange(self.vocab, device=self.omega.device,
                         dtype=torch.float32).unsqueeze(-1) * self.omega   # (V, n_oct)
        code = torch.cat([fip_wave(z, GAMMA, BETA, 0.0),
                          fip_wave(z, GAMMA, BETA, 1.0)], dim=-1)         # (V, 2*n_oct)
        return fip_wave(F.linear(code, self.lift.weight, self.lift_bias),
                        GAMMA, BETA)                                      # (V, D)

    def features(self, tokens):
        # tokens: (B, N) integer char ids
        h = self.char_lut()[tokens]                       # (B, N, D) gather
        B = h.shape[0]
        h = h.reshape(B, -1)
        z = F.linear(h, self.flat_proj.weight, self.flat_bias)
        h = self._phi(z)
        for block in self.blocks:
            h = block(h)
        return h

    def loss(self, tokens, targets, mode=None):
        return self.decoder.loss(self.features(tokens), targets,
                                 mode=mode or self.loss_mode)

    def predict(self, tokens):
        return self.decoder.decode(self.features(tokens))


class MlpCharLM(nn.Module):
    """Classic baseline: embedding table + window flatten + ReLU MLP +
    linear V-dim head with cross-entropy."""

    def __init__(self, vocab, window=32, dim=128, depth=6):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.flat_proj = nn.Linear(window * dim, dim)
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.head = nn.Linear(dim, vocab)

    def features(self, tokens):
        h = self.embed(tokens)
        h = F.relu(self.flat_proj(h.reshape(h.shape[0], -1)))
        for layer in self.layers:
            h = F.relu(layer(h))
        return h

    def loss(self, tokens, targets, mode=None):
        return F.cross_entropy(self.head(self.features(tokens)), targets)

    def predict(self, tokens):
        return self.head(self.features(tokens)).argmax(dim=-1)


# Generic window LM over any discrete vocab (chars, BPE tokens)
FIPWindowLM = FIPCharLM
MlpWindowLM = MlpCharLM


class MlpBaseline(nn.Module):
    """Classic baseline: linear input, ReLU MLP, linear head, cross-entropy."""

    def __init__(self, n_classes=10, dim=128, depth=6):
        super().__init__()
        self.input_proj = nn.Linear(28 * 28, dim)
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.head = nn.Linear(dim, n_classes)

    def features(self, images):
        h = F.relu(self.input_proj(images.flatten(1)))
        for layer in self.layers:
            h = F.relu(layer(h))
        return h

    def loss(self, images, targets):
        return F.cross_entropy(self.head(self.features(images)), targets)

    def predict(self, images):
        return self.head(self.features(images)).argmax(dim=-1)
