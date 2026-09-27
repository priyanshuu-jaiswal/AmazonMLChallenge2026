# Team Contribution Rules

`main` is the stable integration branch. Never work directly on `main`; create a task branch, for example `feature/rare-key-retrieval`, `fix/output-validation`, or `experiment/submission1-reproduction`.

1. Pull or rebase from `main` before beginning work and before opening a PR when appropriate.
2. Keep commits focused and descriptive.
3. Never commit datasets, generated submissions, SQLite indexes/databases, secrets, credentials, API keys, passwords, or huge temporary files.
4. Never modify official source data under `student_resource/dataset/`.
5. Preserve reproducibility by recording seeds, splits/samples, configurations, and commands needed to reproduce results.
6. Give each experiment its own directory under `experiments/`.
7. Record each experiment’s objective, dataset/sampling used, method, parameters, results, interpretation, runtime, and limitations.
8. Do not silently change the frozen Stage 4 candidate-generation design. If a methodology change is proposed, explain why and create a reproducible experiment first.
9. Do not overwrite historical experiment results.
10. Push completed work to a dedicated feature branch and use a PR to merge into `main`.
11. Never force-push shared branches or rewrite shared history.
12. PRs should explain what changed and how it was validated.
13. Run relevant validation/tests before merging and record the result.
