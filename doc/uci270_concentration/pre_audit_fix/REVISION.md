# Pre-fit audit-record revision

Before any real-data model fit, independent review identified that inner-fold
records retained SSE and preprocessing but omitted validation predictions and
coefficients. The new failing test reproduced the missing `predictions` key.

The adapter now also saves inner validation predictions, standardized
coefficients and the standardized intercept, permitting independent replay and
SSE recomputation. All nine model/runner tests passed afterward. This changes
audit records only; data, splits, grids, fitting and selection rules are unchanged.

This directory preserves the preceding contract, configuration, protocol and
model adapter. The parent directory contains the freshly frozen active contract.
