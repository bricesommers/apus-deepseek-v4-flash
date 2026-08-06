
## 2026-08-05 — multi-repo structure: apus-deepseek-v4-flash is the public home

User's plan: one GitHub repo per MoE model (this one first, others in the
future). New empty repo created by the user:
github.com/bricesommers/apus-deepseek-v4-flash — remote `dsv4` added
locally; README clone URL updated; public-release pushed there as `main`
(clean history from the start: the STATUS.md slip never touched it; CI
runs there). Old repo github.com/bricesommers/apus (remote `github`):
its pushed tip still contains the pre-fix commit with STATUS.md in
docs/ (private repo, only the user can see it) — fix with
`git push --force github public-release:main`, or delete the old repo
(user decision: keep it as private umbrella/dev home, or remove).
