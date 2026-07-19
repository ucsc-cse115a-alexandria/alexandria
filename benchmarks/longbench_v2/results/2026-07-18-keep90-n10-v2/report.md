| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 68,824.7 | 0.0% | 0.0000 | 60.0% (6/10) | 14.8s | $0.6371 | 0.0s | $0.0000 |
| keep90 | 61,042.0 | 11.3% | 0.0091 | 60.0% (6/10) | 12.6s | $0.6021 | 290.0s | $0.2925 |

## Release decisions

- **keep90:** PASS: accuracy-retention confidence interval clears the release threshold
