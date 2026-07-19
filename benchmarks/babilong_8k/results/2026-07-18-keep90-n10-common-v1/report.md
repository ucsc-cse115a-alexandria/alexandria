| Condition | Mean input tokens | Token reduction | Cosine difference | Accuracy | Estimated API cost | Wall-clock time |
|---|---:|---:|---:|---:|---:|---:|
| original | 7,668.5 | 0.0% | 0.0000 | 60.0% (6/10) | $0.0768 | 11.5s |
| keep90 | 6,634.2 | 13.5% | 0.0070 | 60.0% (6/10) | $0.2832 | 104.0s |

## Release decisions

- **keep90:** FAIL: accuracy-retention confidence interval does not clear the release threshold
