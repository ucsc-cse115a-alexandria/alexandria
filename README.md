# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/ucsc-cse115a-alexandria/alexandria/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                     |    Stmts |     Miss |   Branch |   BrPart |     Cover |   Missing |
|----------------------------------------- | -------: | -------: | -------: | -------: | --------: | --------: |
| src/alexandria/\_\_main\_\_.py           |        1 |        1 |        0 |        0 |      0.0% |         1 |
| src/alexandria/cli/browser\_review.py    |      144 |       22 |       34 |        7 |     82.6% |37, 55, 59, 133, 163-170, 174-175, 180-182, 201-202, 216-217, 219 |
| src/alexandria/cli/envelope.py           |       22 |        0 |        0 |        0 |    100.0% |           |
| src/alexandria/cli/hunks.py              |       25 |        0 |       12 |        0 |    100.0% |           |
| src/alexandria/cli/interactive.py        |      127 |        6 |       38 |        6 |     92.7% |37, 39, 111, 156-157, 186, 197-\>181 |
| src/alexandria/cli/main.py               |      229 |        2 |       60 |        2 |     98.6% |  134, 403 |
| src/alexandria/cli/review\_html.py       |       68 |        3 |       12 |        1 |     92.5% |   333-336 |
| src/alexandria/cli/verbose.py            |       32 |        0 |        8 |        0 |    100.0% |           |
| src/alexandria/ir/contracts.py           |      169 |        1 |       38 |       18 |     90.8% |73-\>75, 73-\>exit, 75-\>exit, 85-\>exit, 92-\>exit, 138, 231-\>exit, 248-\>250, 248-\>exit, 250-\>252, 250-\>exit, 252-\>254, 252-\>exit, 254-\>261, 254-\>exit, 261-\>exit, 292-\>exit, 312-\>exit |
| src/alexandria/ir/document.py            |      119 |        5 |       36 |        5 |     93.5% |24, 53, 55, 116, 155 |
| src/alexandria/ir/registry.py            |       54 |        0 |        4 |        0 |    100.0% |           |
| src/alexandria/ir/similarity.py          |       21 |        0 |        0 |        0 |    100.0% |           |
| src/alexandria/ops/features/compare.py   |       34 |        0 |        2 |        0 |    100.0% |           |
| src/alexandria/ops/features/diff.py      |       19 |        1 |        8 |        1 |     92.6% |        43 |
| src/alexandria/ops/features/optimize.py  |       92 |        2 |       42 |        3 |     96.3% |82-\>61, 108, 124 |
| src/alexandria/ops/features/represent.py |      194 |        0 |       54 |        0 |    100.0% |           |
| src/alexandria/ops/features/score.py     |       44 |        0 |       12 |        1 |     98.2% |   58-\>62 |
| src/alexandria/ops/features/select.py    |       44 |        2 |       16 |        3 |     91.7% |29, 49, 89-\>87 |
| src/alexandria/ops/features/target.py    |      361 |       51 |      130 |       25 |     80.9% |122-125, 133, 146, 161, 202-212, 218-221, 289-\>301, 300, 319-\>329, 326, 351, 361, 382-383, 387, 388-\>390, 404-419, 430-434, 445, 449-453, 472, 477, 486, 502, 514, 563, 602-\>604, 606 |
| src/alexandria/ops/pipe.py               |       79 |        2 |        4 |        0 |     97.6% |     90-91 |
| src/alexandria/ops/report.py             |       97 |        4 |       14 |        4 |     92.8% |143, 145, 149, 196 |
| src/alexandria/utils/config.py           |       35 |        0 |        6 |        0 |    100.0% |           |
| src/alexandria/utils/embedders.py        |       55 |        2 |       10 |        2 |     93.8% |31, 53-\>55, 92 |
| src/alexandria/utils/merger.py           |       33 |        6 |       10 |        0 |     81.4% | 56-67, 79 |
| src/alexandria/utils/tokens.py           |       20 |        1 |       10 |        2 |     90.0% |18, 27-\>32 |
| **TOTAL**                                | **2118** |  **111** |  **560** |   **80** | **92.0%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/ucsc-cse115a-alexandria/alexandria/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/ucsc-cse115a-alexandria/alexandria/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/ucsc-cse115a-alexandria/alexandria/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/ucsc-cse115a-alexandria/alexandria/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fucsc-cse115a-alexandria%2Falexandria%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/ucsc-cse115a-alexandria/alexandria/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.