"""Real-scale two-half distillation: llama-equal teacher -> FIP student.

Staged checkpointing (atomic, tmp + os.replace):
  stage "p1"  - after task-aware P fit (P + teacher_proj_norm)
  stage "s1"  - after compressor distillation (model + P)
  stage "s2"  - every save_every fry steps (model + P + fry opt state + it)
  stage "done" - after polish + final val acc
Any restart (fix, kill, OOM) resumes from the last stage boundary.

Usage: python experiments/run_distill.py --depth {8,14} [--scale 1.0]
"""
import argparse, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.attention_v3 as av
from experiments.mini_llama import MiniLlama
from experiments.tiny_stories import load_chunks
import experiments.distill as D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--n-attn", type=int, default=1)
    ap.add_argument("--comp-blocks", type=int, default=None)
    ap.add_argument("--teacher", default=r"C:\Users\ignat\AppData\Local\Temp\ck_equal.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--protocol", default="v3", choices=["v3", "v2"])
    args = ap.parse_args()
    device = "cuda"
    out = args.out or rf"C:\Users\ignat\AppData\Local\Temp\ck_distill{args.depth}.pt"
    torch.manual_seed(0)
    train, val = load_chunks(device, 60_000_000, 1_000_000)
    W = 64

    teacher = MiniLlama(dim=160, layers=5, heads=4, allpos=True).to(device)
    st = torch.load(args.teacher, map_location=device)
    teacher.load_state_dict(st["model"])
    print(f"teacher: {args.teacher} step {st['step']}", flush=True)

    torch.manual_seed(0)
    n_attn, comp = args.n_attn, args.comp_blocks or args.n_attn
    assert args.depth >= n_attn, "depth must cover all attention layers"
    # v2 protocol: END-TO-END graph on ALL window positions (proven by the
    # 38.2%/41.4% from-scratch baselines); no latent split, no softvec.
    S = av.AttLM(window=W, input_mode="byte", train_allpos=True, pos_mode="rope",
                 block_kind="gated", dim=args.dim, heads=1, n_attn=n_attn,
                 depth_after=args.depth, code_dim=32).to(device)
    S.decoder.codes(cache_path="data/codebook_2048_fourier32.pt")
    n_par = sum(p.numel() for p in S.parameters())
    print(f"student: dim {args.dim} depth {args.depth} n_attn {n_attn} comp {comp} "
          f"| {n_par/1e6:.2f}M params", flush=True)

    def val_acc():
        with torch.no_grad():
            pos = torch.randint(0, val.shape[0] - W - 1, (512,), device=device)
            off = torch.arange(W, device=device)
            idx = pos.unsqueeze(1) + off.unsqueeze(0)
            x = val[idx].long()
            y = val[idx + 1].long()
            a = (S.predict(x) == y).float().mean().item()   # acc(all), allpos
        torch.cuda.empty_cache()
        return f"val acc(all) {a*100:.2f}%"

    dist = D.ResonanceDistiller(teacher, S, device=device, n_comp_blocks=comp)

    # ---- resume state
    stage, fry_it, fry_opt = None, 0, None
    if os.path.exists(out):
        try:
            ck = torch.load(out, map_location=device)
            stage = ck["stage"]
            S.load_state_dict(ck["model"])
            dist.P = ck["P"].to(device)
            dist.teacher_proj_norm = ck["proj_norm"]
            dist.head_w = D.teacher_logits_w(teacher).detach().to(device)
            if stage == "s2":
                fry_it = ck["fry_it"]
                fry_opt = ck["fry_opt"]
            print(f"resumed: stage {stage}" + (f" at fry {fry_it}" if stage == "s2" else ""),
                  flush=True)
        except (RuntimeError, OSError, EOFError) as e:
            print(f"ckpt unreadable ({type(e).__name__}), fresh", flush=True)
            stage = None

    t0 = time.time()

    def save(stage_name, fry_it_=None, fry_opt_=None):
        tmp = out + ".tmp"
        torch.save({"stage": stage_name, "model": S.state_dict(),
                    "P": dist.P.cpu(), "proj_norm": dist.teacher_proj_norm,
                    "fry_it": fry_it_, "fry_opt": fry_opt_}, tmp)
        os.replace(tmp, out)

    if stage is None:
        D.align_codebook(S, D.teacher_logits_w(teacher).detach(), steps=1500,
                         device=device)
    if stage is None:
        dist.distill_compressor(train, steps=int(6000 * args.scale), lr=2e-3)
        save("s1")
        print(f"[{time.time()-t0:.0f}s] stage1 done | {val_acc()}", flush=True)
        torch.cuda.empty_cache()
    elif stage == "s1":
        print(f"[{time.time()-t0:.0f}s] stage1 SKIPPED (checkpoint) | {val_acc()}",
              flush=True)

    # v2: skip the split stages entirely
    if stage is None:
        save("e2e", 0, None)
        stage = "e2e"
    if stage in ("s1", "s2"):      # legacy checkpoint: treat as e2e start
        stage = "e2e"

    def fry_hook(it, model, opt):
        save("s2", it, {k: v for k, v in opt.state_dict().items()})

    if stage == "e2e" and fry_it < int(24000 * args.scale):
        if args.protocol == "v3":
            D.train_e2e_v3(S, teacher, train, steps=int(24000 * args.scale),
                           batch=256, lr=3e-4, T_kd=4.0, kd_weight=1.0,
                           kd_every=2, kd_until_frac=0.7, eval_fn=val_acc,
                           start_it=fry_it, opt_state=fry_opt,
                           save_hook=fry_hook, save_every=500)
        else:
            D.train_e2e_topk(S, teacher, train, steps=int(8000 * args.scale),
                             batch=256, lr=3e-4, K=32, T_kd=2.0, ce_weight=0.1,
                             eval_fn=val_acc, start_it=fry_it,
                             opt_state=fry_opt, save_hook=fry_hook,
                             save_every=500)
        print(f"[{time.time()-t0:.0f}s] e2e done | {val_acc()}", flush=True)
        torch.cuda.empty_cache()
    else:
        print(f"[{time.time()-t0:.0f}s] e2e SKIPPED (checkpoint: {stage} "
              f"@ {fry_it}) | {val_acc()}", flush=True)

    if stage != "done":
        D.polish(S, teacher, dist.P, train, steps=int(2000 * args.scale),
                 lr=3e-5, device=device, eval_fn=val_acc)
    acc = val_acc()
    print(f"DISTILL depth{args.depth}: {acc} | params {n_par/1e6:.2f}M | "
          f"[{time.time()-t0:.0f}s]", flush=True)
    save("done")

if __name__ == "__main__":
    main()
