# Data card

UCI Concrete Compressive Strength: 1,030 historical measured examples, eight input features and compressive strength in MPa. Ingredients are measured in kg/m³; age is measured in days. Collapsing identical feature vectors yields 996 unique feature rows; reported targets for those rows are averages when repeats exist.

## Source and citation

[Original dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository. https://doi.org/10.24432/C5PK67

## License

UCI lists the dataset under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). Credit the creators and source when reusing data or derived figures. The repository's MIT license applies to its code, not to the dataset.

## Integrity and storage

```text
{
  "url": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
  "sha256": "dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7"
}
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

Rows with identical eight-feature vectors are collapsed to one mean target, preventing exact feature duplicates across splits. The 996 resulting feature rows are split 60%/20%/20% into training/calibration/test. Five-fold training-only CV selects the lowest-MAE model among mean, scaled ridge and histogram boosting. Calibration examples are never fitted. Their absolute residuals define the finite-sample order statistic at rank ceil((n+1)×0.90). Test intervals are evaluated once with that fixed radius. The interval is marginal under exchangeability assumptions, not a per-mixture guarantee.

## Representation and privacy

This is a portfolio service on historical lab data and cannot approve structural designs, material specifications or safety decisions. Conformal coverage assumes exchangeability and does not survive arbitrary shift. Related recipes may cross splits. Monitoring reports training-range violations and standardized mean shifts; it does not detect accuracy degradation or establish a statistically significant distribution change. Authentication, rate limits, TLS, durable telemetry and a model registry are deployment responsibilities. The included container definition is provided; Docker was unavailable for local execution.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.
