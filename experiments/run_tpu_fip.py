"""Kaggle TPU launcher for FIP-Net. One cell, no subprocess probe.

    !python experiments/run_tpu_fip.py --steps 20          (single core)
    !python experiments/run_tpu_fip.py --tpu-parallel ...  (8 cores)

The --tpu-parallel path MUST run in a process that has never initialized
PJRT ("Runtime is already initialized" abort otherwise), so this launcher
never touches a device on that path. Single-core probes in-kernel, where
the answer is authoritative. Pattern from the proven giant-ghl run_tpu.py.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("PJRT_DEVICE", "TPU")

REPO = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(REPO)          # parent of experiments/ - what
if ROOT not in sys.path:              # "import experiments.X" resolves against
    sys.path.insert(0, ROOT)
if REPO not in sys.path:
    sys.path.insert(1, REPO)


def main(argv=None) -> int:
    import experiments.tpu_fip_train as T
    a = T.build_parser().parse_args(argv)
    # fresh session? data is gitignored - say so LOUDLY on stdout
    for f in (a.fit, a.val, a.bytes_mat):
        if not os.path.exists(f):
            msg = ("MISSING: " + f + " - run prepare_wikitext.py and the "
                   "codebook prep cell first (data/ is gitignored)")
            print("*** " + msg + " ***", flush=True)
            return 1
    # fresh session? data is gitignored - say so LOUDLY on stdout
    for f in (a.fit, a.val, a.bytes_mat):
        if not os.path.exists(f):
            print(f"
*** MISSING: {f} - run prepare_wikitext.py + the "
                  f"codebook prep cell first (data/ is gitignored). ***
",
                  flush=True)
            return 1
    # smoke defaults for a first touch of the hardware
    if a.steps == 20000 and a.batch == 1024 and "--steps" not in (argv or []):
        a.steps, a.batch = 20, 8
        a.eval_every, a.eval_windows = 10, 32
        a.save = "/kaggle/working/ck_smoke_fip.pt"
        a.log = "/kaggle/working/runs/fip_tpu_smoke.jsonl"
    return T.main(argv if argv is not None else _to_argv(a))


def _to_argv(a) -> list:
    out = []
    for k, v in sorted(vars(a).items()):
        flag = "--" + k.replace("_", "-")
        if v is True:
            out.append(flag)
        elif v is not False and v is not None and v != "":
            out.append(flag)
            out.append(str(v))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
