# Repository Coverage



| Name                                |    Stmts |     Miss |   Branch |   BrPart |     Cover |   Missing |
|------------------------------------ | -------: | -------: | -------: | -------: | --------: | --------: |
| src/alexandria/\_\_main\_\_.py      |        1 |        1 |        0 |        0 |      0.0% |         1 |
| src/alexandria/cli.py               |       38 |        0 |        0 |        0 |    100.0% |           |
| src/alexandria/core/apply.py        |       32 |        0 |       12 |        0 |    100.0% |           |
| src/alexandria/core/ir.py           |       70 |        3 |       14 |        3 |     92.9% |19, 95, 109 |
| src/alexandria/core/protocols.py    |       35 |        0 |       10 |        6 |     86.7% |34-\>36, 34-\>exit, 36-\>exit, 57-\>exit, 61-\>exit, 65-\>exit |
| src/alexandria/core/registry.py     |       54 |        0 |        4 |        0 |    100.0% |           |
| src/alexandria/core/similarity.py   |       11 |        0 |        0 |        0 |    100.0% |           |
| src/alexandria/phases/optimize.py   |       36 |        1 |        8 |        1 |     95.5% |        42 |
| src/alexandria/phases/represent.py  |      138 |        1 |       36 |        1 |     98.9% |       219 |
| src/alexandria/phases/score.py      |       31 |        0 |        8 |        0 |    100.0% |           |
| src/alexandria/phases/select.py     |       23 |        1 |        6 |        1 |     93.1% |        26 |
| src/alexandria/runtime/embedding.py |       28 |        0 |        2 |        1 |     96.7% | 26-\>exit |
| src/alexandria/runtime/pipeline.py  |       32 |        0 |        6 |        1 |     97.4% |   45-\>49 |
| **TOTAL**                           |  **529** |    **7** |  **106** |   **14** | **96.7%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://github.com/ucsc-cse115a-alexandria/alexandria/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/ucsc-cse115a-alexandria/alexandria/tree/python-coverage-comment-action-data)

This is the one to use if your repository is private or if you don't want to customize anything.



## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.