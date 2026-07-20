# AGENTS.md — Platform & Developer Infrastructure

You are a coding agent working in the Platform / Developer Infrastructure team's monorepo.
This team owns CI/CD, the internal build system, shared libraries, and the deployment tooling other
engineers depend on. Changes here affect every team, so correctness and clarity come before speed.

## Scope of changes
Keep each change focused on one concern; do not mix unrelated fixes into a single change.
Do not bundle unrelated refactors together with a feature or bug fix in the same pull request.
When you find an unrelated problem, note it separately instead of fixing it in the current branch.
Prefer small, reviewable pull requests over large ones that touch many packages at once.
Large sweeping pull requests are hard to review, so split them into smaller ones.

## Testing
Every behavior change must come with a test that would fail without the change.
Write a test that fails before the fix and passes after it.
Do not mark a change complete until the full test suite passes locally.
Run the whole test suite before you open a pull request, and make sure it is green.
Prefer fast, deterministic unit tests over slow tests that reach the network.
Avoid tests that depend on live network calls or external services; inject fakes instead.

## Style and tooling
Follow the repository's configured formatter and linter; do not hand-format code.
Let the automated formatter own layout instead of formatting code by hand.
Never disable a lint rule with an inline ignore comment just to make the check pass.
Do not suppress type-checker or linter errors; fix the underlying cause instead.
Name things for what they are, using the terms already established in this codebase.
Reuse existing names and conventions rather than inventing new ones for the same idea.

## Reliability
Never log secrets, tokens, or credentials, even at debug level.
Keep secrets and credentials out of logs and error messages entirely.
Fail loudly on unexpected errors rather than swallowing them silently.
Do not catch an exception only to hide it; surface it or handle it deliberately.
Make deployment scripts idempotent so re-running them is always safe.
A deploy step must be safe to run more than once without causing damage.

## Review and merge
Do not merge a pull request until continuous integration is fully green.
Wait for every CI check to pass before merging anything to the main branch.
Get at least one review approval before merging a change that touches shared infrastructure.
Shared-infrastructure changes need a second engineer's approval before they merge.
Write a clear pull request description that explains what changed and why.
Describe the motivation and the effect of the change in the pull request body.
