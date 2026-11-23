"""MNIST experiment for FIP-Net variants.

The whole dataset lives on the GPU as uint8 (47 MB); batches are slices,
so there is no DataLoader, no CPU->GPU transfer per step, and no
allocator growth from varying shapes.

Variants:
  fip-harmonic-parabolic  : histogram x LUT harmonic input, parabolic blocks
  fip-harmonic-triangular : histogram x LUT harmonic input, triangular blocks
  fip-linear-parabolic    : linear input projection + parabolic blocks
  fip-linear-triangular   : linear input projection + triangular blocks
  mlp-baseline            : ReLU MLP + cross-entropy (reference)

Usage: python experiments/mnist.py [--epochs 5] [--variants name1,name2]
"""

import argparse
import sys
import time

import torch
from torchvision import datasets

sys.path.insert(0, ".")

from fipnet.model import FIPMnist, MlpBaseline
from fipnet.activation import GAMMA, BETA

VARIANTS = {
    "fip-linear-parabolic-mse": dict(model="fip", input_mode="linear", activation="parabolic", loss_mode="mse", dim=256),
    "fip-linear-parabolic-ce": dict(model="fip", input_mode="linear", activation="parabolic", loss_mode="ce", dim=256),
    "mlp-baseline": dict(model="mlp", dim=128),
}


def load_mnist(device, data_dir="./data"):
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)
    Xtr = train.data.to(device)          # uint8 (60000, 28, 28)
    Ytr = train.targets.to(device)
    Xte = test.data.to(device)
    Yte = test.targets.to(device)
    return Xtr, Ytr, Xte, Yte


def normalize(x_uint8):
    return x_uint8.float().div_(255.0).unsqueeze(1)


@torch.no_grad()
def evaluate(model, Xte, Yte, batch=4096):
    model.eval()
    correct = 0
    for i in range(0, Xte.shape[0], batch):
        idx = slice(i, i + batch)
        correct += (model.predict(normalize(Xte[idx])) == Yte[idx]).sum().item()
    model.train()
    return correct / Xte.shape[0]


def train_variant(name, cfg, epochs, Xtr, Ytr, Xte, Yte, device):
    torch.manual_seed(0)
    if cfg["model"] == "fip":
        model = FIPMnist(n_classes=10, dim=cfg.get("dim", 128), depth=6, code_dim=8,
                         gamma=GAMMA, beta=BETA, input_mode=cfg["input_mode"],
                         activation=cfg["activation"],
                         loss_mode=cfg.get("loss_mode", "ce")).to(device)
    else:
        model = MlpBaseline(n_classes=10, dim=cfg.get("dim", 128), depth=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # plain Adam: verified 97% config
    n = Xtr.shape[0]
    bs = 256
    torch.cuda.reset_peak_memory_stats()
    max_grad_norm = 0.0
    t0 = time.time()
    steps = 0
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            images = normalize(Xtr[idx])
            opt.zero_grad(set_to_none=True)
            loss = model.loss(images, Ytr[idx])
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
            max_grad_norm = max(max_grad_norm, gnorm)
            opt.step()
            steps += 1
        acc = evaluate(model, Xte, Yte)
        best_acc = max(best_acc, acc)
        print(f"  [{name}] epoch {epoch}: test acc {acc*100:.2f}%  "
              f"(max|grad| so far {max_grad_norm:.1f})", flush=True)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    allocated = torch.cuda.memory_allocated() / 2**20
    final = evaluate(model, Xte, Yte)
    n_params = sum(p.numel() for p in model.parameters())
    return dict(name=name, acc=final, best_acc=best_acc, params=n_params,
                seconds=dt, peak=peak, allocated=allocated, steps=steps,
                max_grad_norm=max_grad_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--variants", type=str, default=None,
                    help="comma-separated subset of variant names")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    Xtr, Ytr, Xte, Yte = load_mnist(device)
    names = args.variants.split(",") if args.variants else list(VARIANTS)
    results = []
    for name in names:
        print(f"training {name} ...", flush=True)
        results.append(train_variant(name, VARIANTS[name], args.epochs,
                                     Xtr, Ytr, Xte, Yte, device))
    print("\n=== MNIST summary (test accuracy) ===")
    print(f"{'variant':28s} {'best %':>7s} {'final %':>8s} {'params':>9s} {'sec':>7s} "
          f"{'ms/step':>8s} {'peak MB':>8s} {'max|grad|':>10s}")
    for r in results:
        print(f"{r['name']:28s} {r['best_acc']*100:7.2f} {r['acc']*100:8.2f} {r['params']:9d} "
              f"{r['seconds']:7.1f} {r['seconds']/r['steps']*1000:8.2f} "
              f"{r['peak']:8.0f} {r['max_grad_norm']:10.1f}")


if __name__ == "__main__":
    main()
