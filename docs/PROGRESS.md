# PROGRESS / HANDOFF (updated: 2026-08-24 ~16:45)

## Where we are NOW
RH v2 redesign COMPLETE and verified end-to-end (two-arm firing +
prior + fit-quality gate + certified-step probe). check_v2 9/9 OK,
theory_check 13/13 PASS. RH now wins or ties on ALL FOUR families
(quadratic 29 its, logistic 30 its, logsumexp 156 its/0.2s -- was
827k-its FAILURE, rosenbrock gn=0.00e+00 vs NAG-AR 34272 its).
Purge #4 applied; supervisor RUNNING (E6+E8 alternation -> analysis.py).

## Status snapshot (2026-08-24 ~16:45)
- hybrid_v2.py `rate_hybrid`: NEW SEMANTICS (see Ledger #4). Params:
  win=30 span_trust=0.5 check_every=10 margin=1.5 cap_frac=0.5
  c2_prior=100.0 streak_need=5 backoff_max=200 fo_mode=auto|bb verbose.
- run_all_v2.py: e8_methods now includes "RH-BB" (fo_mode="bb" ablation).
- tests/check_v2.py: ALL OK (incl. logsumexp convergence; AR-vs-BB test
  is now informational -- neither FO scheme dominates post-redesign).
- tests/theory_check.py: 13/13 PASS (incl. new [regime] + [basin]).
- THEORY.md: regime map + Lemma 3b added; section 5 contract = steps
  1-8 matching code exactly.
- all_runs_v2.csv: purged to 1367 rows (E6=66 E7=601 E8=700), zero RH.
  E7 complete (601 rows; 30 theta=1e-8 conv=0 BY DESIGN, KEEP).
- Supervisor: supervisor_e6e8.ps1 IN PROJECT ROOT (NOT %TEMP% -- an
  external process wipes %TEMP%\opencode periodically; old supervisor
  died that way). Logs results\logs\. todo at launch: E6=294 E8=200.

## Issue 1 CLOSED -- Newton-CG (final design in methods.py newton_cg)
Two-layer root cause, each layer fixed separately:

Layer A (inner CG): EW forcing eta=min(1e-4, gn0^1.5) is UNREACHABLE at
kappa=1e4 within product cap min(n,1000) -- Krylov wall
((sqrt(kappa)-1)/(sqrt(kappa)+1))^k ~= e^-2 after 100 products.
Fix: every (k+1)%25==0 compute TRUE residual rt=b-hv(x,p) (+1 counted
matvec), track p_best by true rel residual, break if rel<=eta.
CRITICAL: MONITOR-ONLY. Do NOT feed rt back into the recursion
(r<-rt breaks conjugacy; measured slowdown 18 vs 8 outer its).

Layer B (outer acceptance): plain Armijo-on-f COLLAPSES t to micro-steps
for inexact CG directions: actual quadratic decrease (-1/2 g'p - 1/2 p'r)
undershoots the slope prediction c1*t*g'p when p'r>0 (observed: a down to
5.8e-11, gn frozen at 4.13e-6). Pure norm-decrease test fixes quadratics
(full step t=1 accepted since g+ == r_true exactly on quadratics) BUT
FREEZES on Rosenbrock saddle pockets: gradient can align with a
NEGATIVE-curvature direction so ||grad|| increases along -pg at EVERY
representable t (measured: phi rises even at t=2^-50, local slope ~ -2400;
CG k=0 curvature test fires neg0 -> p=zeros forever).
FINAL outer loop = ordered TWO-STAGE test per trial t:
  1. strict norm decrease: ||g(x+t*p)|| < (1-1e-12)*||g||
  2. else Armijo-on-f: f(x+t*p) <= f_x + 1e-4 * t * (g'p)
  SD fallback: classic _backtrack Armijo, micro-step 1e-6/gn guarantee.
Results: kappa1e4-n100 s1..9 -> it=8 (was 2000); kappa1e4-n1000 -> it=4
~1-2s (was 120s time-cap); rosenbrock n50/n100 -> it=93/166 (was frozen).

## Issue 2 CLOSED -- RH rosenbrock tail crawl
rate_hybrid rewritten as single-loop mode machine ('fo'|'tail'):
- probe success -> mode='tail', switched_at recorded once.
- tail stagnation monitor: every 50 tail its, bail if gn > 0.98*ref_gn
  (ref_gn = min anchored AT tail entry, refreshed via min() at checks --
  literal "gn>0.98*ref_gn" self-triggers when gn IS the running min).
- bail: giveups+=1, cooldown_until=it1+min(50*2**min(giveups,9),500),
  win_gn=[], momentum reset (y_old=x_cur, anag.px=None), fresh grad+f.
- EXTRA guard added: micro-probe a<1e-6 => probes_failed++,
  cooldown min(5*2**min(probes_failed,8), backoff_max).
Result: rosenbrock n100 conv=1 sw=True p1=37721 p2=200 giveups=1
t=9-21s gn=9.94e-09 (was it=20007 conv=0).

## Test-suite metric fix (check_v2.py)
Old assertion compared FINAL gnorms of two CONVERGED runs (meaningless
overshoot comparison -> spurious FAIL). New fair metric: iterations each
FO scheme needs to reach the SAME target level (BB's switched_at_gnorm):
AR 5066 <= BB p1 6230 -> PASS. Keep this metric.

## Purge ledger (all applied to all_runs_v2.csv + hist_v2 npz)
1. Backup -> .pre_semantics.bak. Dedupe keys (turned out none dup'd;
   1405 unique). Purged 202 rows whose method semantics changed:
   E8/RH=100, E8/Newton-CG=100, E6 smoke RH+Newton-CG=2 (smoke rows would
   have BLOCKED real E6 reruns via resume-skip!).
2. +20 frozen E8 Newton-CG rosenbrock rows (mid-session failure, then
   fixed by Layer-B Armijo fallback).
3. +1 orphan E8 row logistic lam0.01-m3000-n500 NAG-CM seed=0 (cfg not in
   current E8 grid; would pollute analysis as lone cell).
4. 2026-08-24 ~16:30: RH SEMANTICS REDESIGN -> purged ALL RH rows
   (E8=100, E6=6) + 106 hist_v2 *__RH____*.npz. Backup ->
   results/all_runs_v2.csv.pre_rh_semantics.bak (=1367+106=1473 rows).
   New RH semantics (all verified by check_v2 + theory_check):
   a. Pessimistic prior c2_eff = c2_prior*c1_hat until first measured
      probe (lateness cheap per Lemma-3 regime map; earliness cost 6x).
   b. Trust gate: FULL window AND 0<rho_hat<1 AND span>=span_trust(0.5)
      AND resid<=max(0.15*span,0.05). Span is SNR-not-progress: old
      span>=3.0 gate was only satisfiable via RISING transients
      (E8 data: rho_hat=1.39>1 at switch) -- anti-correlated with
      switch-worthiness.
   c. Two-arm firing. Arm A validated economics as before. Arm B
      DESPERATION MEASUREMENT: not-trusted but cap_streak>=5 under
      theta_cap => probe anyway to MEASURE c2_hat (slow-flat regimes
      where no window ever looks like clean exponential descent).
   d. Probe acceptance TWO-STAGE mirroring the tail: strict
      gradient-NORM decrease (float-floor-safe), else Armijo on f.
      A probe fails ONLY if neither arm certifies any step (a<1e-12);
      tiny-but-certified steps ARE accepted -- near-flat curvature
      gives correct Newton directions at pathological lengths
      (logsumexp a~9e-10); the tail's stagnation monitor polices
      productivity. This single change took logsumexp from 827k its
      FAIL to 156 its / 0.2s / 1 probe / 0 giveups.
   e. fo_mode="bb" registered as "RH-BB" in E8 (ablation; BB phase 1
      crushes NAG-AR on rosenbrock: p1=129 vs AR-to-target 4972).
Rule going forward: ANY code-semantics change => purge ALL rows of that
method across ALL exps, else mixed-semantics comparisons.

## Pipeline state
- SUPERVISOR RUNNING: supervisor_e6e8.ps1 (project root), pid 14456 at
  launch ~16:42. STRICTLY SEQUENTIAL E6(60-min sessions)/E8(25-min)
  alternation until both todo=0, then runs analysis.py automatically.
  Logs: results\logs\supervisor_trace.log + session_*.log.
- Resume-safe: kill/restart supervisor any time; completed rows skip.
- At launch: E6=294 todo (of 360; now includes new-semantics RH),
  E8=200 todo (=100 RH + 100 RH-BB re-runs, all fast),
  E7 complete todo=0. E6 conv=0-with-t~120s rows LEGITIMATE (wall-clock
  profiling exp, max_time=120s BY DESIGN).
- THEN phase-4 docs: THEORY.md numbers refresh from new tables,
  plan.txt rewrite, REVIEW.md, README.
- PAPERS WRITTEN (2026-08-24 ~17:10): paper/paper_en.tex -> paper_en.pdf
  (7 pp, pdflatex+lmodern), paper/paper_ar.tex -> paper_ar.pdf (7 pp,
  xelatex+polyglossia, font "Arabic Typesetting" Scale=1.28 -- NOT
  "Traditional Arabic", absent on this box). Figures: code/make_paper_figures.py
  -> results/figs/F1..F6 (rerun after todo=0 to refresh RH cells).
  Thesis plan: PLAN_THESIS_AR.md.
- Commits: git -c user.name="jamil-junaidi" -c user.email="jamil@local".
  Last commit: 3c61598. .gitignore covers data/ __pycache__/.

## Debug scripts (rerunnable, %TEMP%\opencode\)
- dbg_targets.py      : targeted verification of formerly-failing cells
- dbg_quad_fix.py     : quadratic sweep post-fix (uses r['hvevals'])
- dbg_rosen_fix.py    : rosenbrock NC-G spot check
- dbg_rosencg.py      : instrumented replica exposing neg0/freeze branches
- dbg_freeze.py       : line-search deep probe at freeze point (proof of
                        negative curvature along gradient)
- dbg_cg_variants.py  : orig vs replace vs monitor inner-CG toy (18v8)
- dbg_cg_outer.py     : policy A backtrack vs policy B full-step
- dbg_rh_metric.py    : fair phase-1 metric investigation
- purge_v2.py / purge_frozen.py : purge scripts (pattern reference)
Quadratic seeding: default_rng(9000+seed).standard_normal(n);
Rosenbrock x0 alternating -1.2/1.0. Stats key for hv counter is
'hvevals' (NOT 'hv').

## Env facts
- Python: C:\Users\Windows.11\AppData\Local\Programs\Python\Python313\python.exe
- PowerShell shell; no rg; prefer writing .py scripts to
  %TEMP%\opencode instead of inline python -c quoting gymnastics
  (PowerShell mangles embedded double quotes in -c strings).
- Background runs: Start-Process -PassThru w/ -RedirectStandardOutput/
  -RedirectStandardError; poll Get-Process -Id; note redirected handles
  can hold the launching shell open past command completion (harmless).
- Working dir: C:\Users\Windows.11\Desktop\hyper
- Data present: data/a9a mushrooms w8a (lambda=1/m convention).
