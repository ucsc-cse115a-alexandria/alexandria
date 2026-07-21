| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,574.2 | 0.0% | 0.0000 | 74.0% (37/50) | 49.5s | $0.0000 | 0.0s | $0.0000 |
| default | 7,514.2 | 0.8% | 0.0025 | 70.0% (35/50) | 59.4s | $0.0000 | 1158.8s | $0.0000 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (74.0% accuracy; minimum 0.0%)

## Release decisions

- **default:** FAIL: accuracy-retention confidence interval does not clear the release threshold
