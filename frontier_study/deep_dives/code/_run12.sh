#!/bin/bash
cd /sessions/zealous-practical-sagan/mnt/marketing_measurement/frontier_study/deep_dives/code
python3 - "$@" <<'PY'
import sys, time
sys.argv=['x']+sys.argv[1:]
src=open('12_surrogacy_horizon_bounds.py').read()
exec(compile(src,'12_surrogacy_horizon_bounds.py','exec'))
which = sys.argv[1:] or ["quick"]
t0=time.time()
fns = {k: v for k, v in globals().items() if k.startswith("e") and callable(v)}
if which == ["quick"]:
    for name in QUICK: fns[name]()
else:
    for w in which:
        for k, f in list(fns.items()):
            if k.startswith(w + "_") or k == w: f()
print(f"[{time.time()-t0:.1f}s]")
PY
