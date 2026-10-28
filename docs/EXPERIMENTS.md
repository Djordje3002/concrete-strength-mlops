# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m concrete_ml train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m concrete_ml predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

Rows with identical eight-feature vectors are collapsed to one mean target, preventing exact feature duplicates across splits. The 996 resulting feature rows are split 60%/20%/20% into training/calibration/test. Five-fold training-only CV selects the lowest-MAE model among mean, scaled ridge and histogram boosting. Calibration examples are never fitted. Their absolute residuals define the finite-sample order statistic at rank ceil((n+1)×0.90). Test intervals are evaluated once with that fixed radius. The interval is marginal under exchangeability assumptions, not a per-mixture guarantee.

## Reading the result

The boosted regressor reaches 3.379 MPa test MAE and R² 0.907 on 200 held-out feature rows. The nominal 90% interval covers 91% of the test targets and is 19.203 MPa wide. Width matters alongside coverage: a high-coverage interval can still be too broad for a decision. Similar mixture families can remain across partitions, so this is not proof of generalization to entirely new recipes or laboratories.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.
