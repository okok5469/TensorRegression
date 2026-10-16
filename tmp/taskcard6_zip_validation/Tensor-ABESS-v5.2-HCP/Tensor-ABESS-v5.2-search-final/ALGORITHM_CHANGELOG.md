# Approved algorithm changes

- v5-original to v5.1-bcd-fix: fixed-size BCD explicitly retains the best visited
  OLS state, with nonincreasing accepted loss; cold exploration is preserved.
- v5.1 to v5.2-search-final: enumerate every integer size triple within the old
  per-mode caps and q <= floor(0.7*n). Positions are still searched heuristically.
- Group-BESS, initialization, caps, public signatures and original MSIC are retained.
  Global MSIC alpha=1.5, gamma=1, one-SE scale=1; mode alpha=1.
- Do not interpret the heuristic one-SE tolerance as measured CV standard error.
- Existing 700 task labels: Exact 138 to 182; mean PMSE 2.236761 to 2.381274.
  The labels contain repeated data and do not represent 700 independent datasets.
  Small samples and weak signals remain material limitations; no universal
  prediction superiority or new optimality theorem is claimed.
