# Mushroom full-fold pilot correctness

Status: PASS. Frozen contract `b45fa5b103bace02d780fe1ce265bc31973c5b2e4c9355b8eab5664dcd8effa5`. All eight models independently replayed.

Verified all artifact hashes, native runtime identity, training X/y scalers, constant masks, response inverse and held-out IDs. M1 fixed-support OLS and original MSIC independently reconstructed. Unexposed convergence remains unknown; full candidate and selected candidate statistics are in PILOT_VALIDATION.json.

Compact cache: 3658492 bytes. Per-model full-fold wall times: M0=1.03s, M1=121.42s, M2=135.70s, M3=79.88s, M4=6.31s, M5=3.95s, M6=4.86s, M7=4.64s.

These are actual full first-fold times; no small-sample cubic extrapolation. Peak process working set is recorded separately by the operational controller.

Actual unchanged200-record M1 fit memory probe: interpreter peak working set 67682304 bytes; sampled launcher+interpreter tree RSS peak 71626752 bytes. Extra fit time 9.991533200000049s; saved predictions agree. This is the M1 outer fit, not a claim about maximum of all tuning models. Two engineering repetitions were performed; first launcher-only sampling was corrected to measure the actual interpreter and descendants. Neither changed model settings or formal artifacts.
