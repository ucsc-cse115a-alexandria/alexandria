| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Estimated API cost | Wall-clock time |
|---|---:|---:|---:|---:|---:|---:|
| original | 6,092.4 | 0.0% | 0.0000 | 70.0% (7/10) | $0.0715 | 14.6s |
| keep90 | 5,214.5 | 14.4% | 0.0236 | 50.0% (5/10) | $0.2105 | 83.5s |

## Release decisions

- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
