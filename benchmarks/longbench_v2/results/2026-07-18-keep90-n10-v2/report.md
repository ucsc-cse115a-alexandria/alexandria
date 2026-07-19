| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Estimated API cost | Wall-clock time |
|---|---:|---:|---:|---:|---:|---:|
| original | 68,824.7 | 0.0% | 0.0000 | 60.0% (6/10) | $0.6371 | 14.8s |
| keep90 | 61,042.0 | 11.3% | 0.0091 | 60.0% (6/10) | $0.8946 | 302.6s |

## Release decisions

- **keep90:** PASS: accuracy-retention confidence interval clears the release threshold
