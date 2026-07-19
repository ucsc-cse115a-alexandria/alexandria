| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Execution time | Execution cost | Reduction time | Reduction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| original | 7,668.5 | 0.0% | 0.0000 | 60.0% (6/10) | 11.5s | $0.0768 | 0.0s | $0.0000 |
| keep90 | 6,634.2 | 13.5% | 0.0070 | 60.0% (6/10) | 9.2s | $0.0665 | 94.8s | $0.2166 |

## Release decisions

- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
