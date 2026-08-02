# Tensor-ABESS validation report

## Deterministic checks

| Case | Passed | Exact rectangular | Frobenius error |
|---|---:|---:|---:|
| singleton_exact | yes | yes | 0.0000 |
| low_noise_rectangular | yes | yes | 0.0090 |

## Monte Carlo

- repetitions: 8
- mean S1 TPR: 0.688
- mean S2 TPR: 0.688
- mean S3 TPR: 0.688
- exact recovery rate: 0.375
- mean Frobenius error: 1.285

Plot: `tensor_abess_validation.png`

Interpretation: higher TPR / exact rate and lower Frobenius error are better.