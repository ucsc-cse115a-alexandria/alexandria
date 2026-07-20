| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 6,508.7 | 0.0% | 0.0000 | 74.0% (37/50) | 69.1s | $0.3675 | 0.0s | $0.0000 |
| keep50 | 2,474.9 | 62.0% | 0.1085 | 30.0% (15/50) | 61.4s | $0.1456 | 660.5s | $2.4918 |
| keep60 | 3,475.8 | 46.6% | 0.0638 | 38.0% (19/50) | 63.5s | $0.2059 | 673.7s | $2.5177 |
| keep70 | 4,236.6 | 34.9% | 0.0480 | 46.0% (23/50) | 68.3s | $0.2512 | 568.8s | $1.9577 |
| keep80 | 4,932.8 | 24.2% | 0.0356 | 48.0% (24/50) | 69.3s | $0.2863 | 449.6s | $1.3373 |
| keep90 | 5,613.6 | 13.8% | 0.0255 | 50.0% (25/50) | 70.8s | $0.3218 | 338.2s | $0.7667 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (74.0% accuracy; minimum 50.0%)

## Release decisions

- **keep50:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep60:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep70:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep80:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
