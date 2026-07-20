| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,574.2 | 0.0% | 0.0000 | 72.0% (36/50) | 65.9s | $0.3787 | 0.0s | $0.0000 |
| keep75 | 5,393.2 | 28.8% | 0.0194 | 48.0% (24/50) | 66.9s | $0.2707 | 896.4s | $2.2665 |
| keep80 | 5,772.5 | 23.8% | 0.0148 | 44.0% (22/50) | 59.4s | $0.2895 | 781.5s | $1.8410 |
| keep85 | 6,166.0 | 18.6% | 0.0109 | 48.0% (24/50) | 61.5s | $0.3091 | 810.3s | $1.4273 |
| keep90 | 6,573.5 | 13.2% | 0.0073 | 64.0% (32/50) | 183.1s | $0.3292 | 601.5s | $1.0208 |
| keep95 | 6,967.9 | 8.0% | 0.0050 | 62.0% (31/50) | 79.5s | $0.3486 | 439.6s | $0.6114 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (72.0% accuracy; minimum 50.0%)

## Release decisions

- **keep75:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep80:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep85:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep95:** FAIL: accuracy-retention confidence interval does not clear the release threshold
