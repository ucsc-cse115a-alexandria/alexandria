| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 6,391.0 | 0.0% | 0.0000 | 100.0% (5/5) | 4.7s | $0.0324 | 0.0s | $0.0000 |
| keep50 | 2,271.6 | 64.5% | 0.1341 | 0.0% (0/5) | 4.4s | $0.0117 | 40.5s | $0.1160 |
| keep60 | 3,153.8 | 50.7% | 0.0640 | 60.0% (3/5) | 4.5s | $0.0163 | 51.4s | $0.1411 |
| keep70 | 4,242.6 | 33.6% | 0.0222 | 40.0% (2/5) | 4.2s | $0.0217 | 52.6s | $0.1252 |
| keep80 | 4,807.2 | 24.8% | 0.0201 | 80.0% (4/5) | 4.6s | $0.0245 | 41.7s | $0.0955 |
| keep90 | 5,581.6 | 12.7% | 0.0129 | 80.0% (4/5) | 4.5s | $0.0285 | 29.4s | $0.0520 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (100.0% accuracy; minimum 50.0%)

## Release decisions

- **keep50:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep60:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep70:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep80:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
