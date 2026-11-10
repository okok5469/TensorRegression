# UCI270 frozen protocol

Six separate gas regressions; dev1 batches 1-6 -> 7, dev2 1-7 -> 8, dev3 1-8 -> 9; final 1-9 -> 10. The most recent up to three eligible inner batch boundaries are used; fewer than two blocks evaluation. Exact dynamic96 duplicates are removed from training against validation/test, using X only. Eight methods use identical cleaned IDs. Input is sensor x smoothing alpha x rising/decaying dynamic features (16 x 3 x 2); M6/M7 alone use all128. Every fold scales its own training X and y; no missing-value imputation or prediction clipping.

M0 mean; M1 frozen native ABESS with original MSIC; M2 validated LEMON grouped-ISTA adapter with rectangular OLS; M3 Lasso96; M4 Ridge96; M5 PLS96; M6 Ridge128; M7 Lasso128. All grids, solver parameters, precision retry and tie tolerance are recorded in config.json. Lasso alpha_max is recomputed per training fold; inner selection minimizes pooled SSE/validation count. PLS uses the intersection of training rank-legal candidates and scale=False. ABESS and GLasso do not expose full convergence certification.

One worker, BLAS/OMP/MKL=1; shared cumulative model subprocess wall time 14400 seconds; each task at most 1800 seconds. Pilot must pass an independent lock before development continuation; final remains sealed until a matching independent development lock. Existing results are never overwritten. No remote submission is authorized by this runner.

Contract SHA256: `246acfaef16d0c6452116d46347958f1d0e645cec7e3e468f8cec6a4d426eccc`. See contract.json for raw hashes, full source dependency list, actual imports, environment and exact row IDs.
