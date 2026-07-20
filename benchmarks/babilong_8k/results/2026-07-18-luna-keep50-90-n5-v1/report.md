| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,686.2 | 0.0% | 0.0000 | 60.0% (3/5) | 4.7s | $0.0111 | 0.0s | $0.0000 |
| keep50 | 3,507.0 | 54.4% | 0.0914 | 20.0% (1/5) | 6.2s | $0.0178 | 114.3s | $0.3944 |
| keep60 | 4,222.0 | 45.1% | 0.0469 | 40.0% (2/5) | 6.8s | $0.0213 | 100.8s | $0.3296 |
| keep70 | 5,021.8 | 34.7% | 0.0261 | 40.0% (2/5) | 4.5s | $0.0253 | 84.4s | $0.2497 |
| keep80 | 5,774.8 | 24.9% | 0.0150 | 40.0% (2/5) | 4.3s | $0.0290 | 71.3s | $0.1881 |
| keep90 | 6,615.0 | 13.9% | 0.0066 | 60.0% (3/5) | 4.7s | $0.0331 | 45.8s | $0.1064 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (60.0% accuracy; minimum 50.0%)

## Release decisions

- **keep50:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep60:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep70:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep80:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
