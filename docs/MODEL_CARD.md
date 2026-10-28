# Model card — Concrete Strength MLOps

Version: 1.0.0 · Experiment seed: 42 · Selected model: `hist_gradient_boosting`

## Intended use

Estimate concrete compressive strength from mixture ingredients and age, then expose the trained model through a documented batch API. The project adds split-conformal intervals, artifact identity, readiness, request validation and descriptive input monitoring to a reproducible regression experiment.

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

Rows with identical eight-feature vectors are collapsed to one mean target, preventing exact feature duplicates across splits. The 996 resulting feature rows are split 60%/20%/20% into training/calibration/test. Five-fold training-only CV selects the lowest-MAE model among mean, scaled ridge and histogram boosting. Calibration examples are never fitted. Their absolute residuals define the finite-sample order statistic at rank ceil((n+1)×0.90). Test intervals are evaluated once with that fixed radius. The interval is marginal under exchangeability assumptions, not a per-mixture guarantee.

| Model | mae | rmse | r2 |
| --- | ---: | ---: | ---: |
| `mean_baseline` | 12.7055 | 15.8658 | -0.0076 |
| `scaled_ridge` | 7.9833 | 9.8284 | 0.6133 |
| `hist_gradient_boosting` | 3.3790 | 4.8306 | 0.9066 |


## Interpretation

The boosted regressor reaches 3.379 MPa test MAE and R² 0.907 on 200 held-out feature rows. The nominal 90% interval covers 91% of the test targets and is 19.203 MPa wide. Width matters alongside coverage: a high-coverage interval can still be too broad for a decision. Similar mixture families can remain across partitions, so this is not proof of generalization to entirely new recipes or laboratories.

## Limitations

This is a portfolio service on historical lab data and cannot approve structural designs, material specifications or safety decisions. Conformal coverage assumes exchangeability and does not survive arbitrary shift. Related recipes may cross splits. Monitoring reports training-range violations and standardized mean shifts; it does not detect accuracy degradation or establish a statistically significant distribution change. Authentication, rate limits, TLS, durable telemetry and a model registry are deployment responsibilities. The included container definition is provided; Docker was unavailable for local execution.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.
