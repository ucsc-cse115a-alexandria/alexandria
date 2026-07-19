| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,574.2 | 0.0% | 0.0000 | 68.0% (34/50) | 47.7s | $0.3787 | 0.0s | $0.0000 |
| keep50 | 3,448.6 | 54.5% | 0.0724 | 14.0% (7/50) | 52.4s | $0.1745 | 1234.1s | $4.0457 |
| keep60 | 4,235.1 | 44.1% | 0.0443 | 34.0% (17/50) | 51.4s | $0.2133 | 1049.7s | $3.4965 |
| keep70 | 4,995.9 | 34.0% | 0.0255 | 30.0% (15/50) | 46.8s | $0.2511 | 834.4s | $2.6547 |
| keep80 | 5,780.7 | 23.7% | 0.0147 | 42.0% (21/50) | 44.8s | $0.2899 | 623.1s | $1.8414 |
| keep90 | 6,570.7 | 13.2% | 0.0074 | 72.0% (36/50) | 46.0s | $0.3290 | 446.5s | $1.0202 |

## Baseline qualification

- **Original:** PASS: original accuracy clears the minimum baseline (68.0% accuracy; minimum 50.0%)

## Release decisions

- **keep50:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep60:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep70:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep80:** FAIL: accuracy-retention confidence interval does not clear the release threshold
- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
