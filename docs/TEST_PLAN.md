# Test Plan

**106 tests across six modules, all passing.** Run with `make test`.

## Strategy

The suite is organised around the four ways this project could silently produce a wrong or
unsafe result:

1. **The data is not what we think it is** → dataset contract tests.
2. **The artifact does not match the training run** → artifact and metadata tests.
3. **The model behaves differently at inference than in training** → prediction tests.
4. **The deployment configuration is wrong or leaks a credential** → deployment tests.

Every test either encodes a documented requirement or guards a specific failure mode that
was considered plausible. None exists to inflate the count.

---

## 1. `tests/test_data_validation.py` — 13 tests

| Test | Guards against |
|---|---|
| `test_raw_dataset_exists` | Working on a missing or moved file |
| `test_schema_columns_exact_and_ordered` | Column drift or reordering |
| `test_row_count` | Truncated or appended file |
| `test_column_count` | Added or removed column |
| `test_customer_id_is_unique` | Duplicated records |
| `test_target_domain` | Unexpected target values |
| `test_senior_citizen_domain` | Domain drift on the binary flag |
| `test_no_duplicate_rows` | Silent duplication |
| `test_categorical_domains_match_data_dictionary` | Categories outside the documented set |
| `test_numeric_columns_have_no_negative_values` | Impossible values |
| `test_total_charges_blanks_are_only_zero_tenure_customers` | The documented blanks becoming arbitrary gaps |
| `test_full_validator_passes` | Regression in the validator itself |
| `test_git_blob_sha_matches_official_file` | A substituted or edited dataset |

The SHA test skips — never fails — when Git is unavailable, because absence of Git is not
evidence of a wrong file. CI runs the validator with `--strict-sha`, where absence *is* a
failure.

## 2. `tests/test_model_artifact.py` — 12 tests

| Test | Guards against |
|---|---|
| `test_model_artifact_exists` | Deploying without training |
| `test_metadata_and_schema_exist` | Partial artifact export |
| `test_pipeline_loads_and_is_a_complete_pipeline` | Saving a bare estimator with preprocessing left outside |
| `test_pipeline_loads_in_a_fresh_process` | An artifact that only works in the training session |
| `test_metadata_contains_real_values` | Missing required metadata keys |
| `test_metadata_metrics_are_numeric_and_in_range` | Placeholder strings such as `"ACTUAL_NUMBER"` |
| `test_metadata_reports_a_real_selected_model` | Template tokens surviving into a shipped artifact |
| `test_feature_schema_excludes_identifier_and_target` | `customerID` or `Churn` leaking into the features |
| `test_feature_schema_order_matches_training_configuration` | Schema/pipeline drift |
| `test_feature_schema_categories_cover_the_dataset` | Empty or invented category lists |
| `test_feature_schema_numeric_bounds_are_non_negative` | Corrupt bounds reaching the UI |
| `test_pipeline_feature_names_exclude_identifier_and_target` | The fitted object disagreeing with the schema |

The fresh-process test spawns a real subprocess. An artifact that only loads inside the
session that produced it is not deployable, and an in-process assertion would not detect it.

## 3. `tests/test_prediction.py` — 11 tests

| Test | Guards against |
|---|---|
| `test_prediction_shape` | Shape errors on batch input |
| `test_predicted_classes_are_binary` | Unexpected class labels |
| `test_probabilities_within_unit_interval` | Out-of-range or non-normalised probabilities |
| `test_single_row_prediction` | The exact path the app uses failing |
| `test_unknown_categorical_value_does_not_crash` | A crash on an unseen category in production |
| `test_missing_numeric_value_is_imputed_not_fatal` | A crash on a missing numeric input |
| `test_input_column_order_is_enforced` | Positional rather than name-based column selection |
| `test_prediction_is_deterministic` | Non-reproducible scoring |
| `test_identifier_column_is_rejected_or_ignored` | `customerID` influencing a score |
| `test_threshold_and_class_agree` | The displayed class disagreeing with the displayed probability |
| `test_risk_tier_boundaries` | Off-by-one errors at band edges |

`test_input_column_order_is_enforced` reverses the column order and requires an identical
probability. Positional selection is a classic silent-corruption bug: the pipeline would
still run and still return a plausible number, just a wrong one.

The smoke fixture is built from five real dataset rows with `Churn` and `customerID`
dropped, so the predictors are genuine and no label can leak into them.

## 4. `tests/test_deployment_files.py` — 16 tests

| Test | Guards against |
|---|---|
| `test_docker_runtime_files_exist` | An incomplete deployment package |
| `test_artifacts_are_present_for_deployment` | Deploying without artifacts |
| `test_app_never_imports_the_training_package` | The app depending on code the image does not contain |
| `test_deployment_requirements_cover_every_runtime_import` | A missing runtime dependency — parsed from the AST, not guessed |
| `test_deployment_requirements_are_pinned` | Unpinned versions drifting between builds |
| `test_runtime_scikit_learn_pin_matches_training_environment` | Unpickling under a different scikit-learn version |
| `test_dockerfile_configuration` | Missing base image, env vars, non-root user, port or flags |
| `test_dockerfile_contains_no_credentials` | A token baked into the image |
| `test_space_readme_metadata_uses_docker_and_port_7860` | A Space that will not start |
| `test_workflows_exist` | Missing CI or deployment configuration |
| `test_deploy_workflow_reads_only_the_expected_secret_and_variable` | Scope creep in credential usage |
| `test_deploy_workflow_configuration` | Missing repo type, SDK, subdirectory, concurrency, permissions or LFS |
| `test_workflows_never_echo_the_token` | A credential reaching CI logs |
| `test_no_secret_patterns_in_project_files` | Any committed credential |
| `test_gitignore_covers_sensitive_paths` | Weakened ignore rules |
| `test_no_env_file_is_committed_to_the_repository` | A `.env` appearing in the tree |

`test_deployment_requirements_cover_every_runtime_import` parses each runtime module's AST
and checks every third-party import against the pinned requirements, mapping import names
to distribution names where they differ (`sklearn` → `scikit-learn`). A hand-maintained
list would go stale; this cannot.

The secret-pattern test reports the file path and category only. A test that printed the
match would leak the secret into CI output.

## 5. `tests/test_analysis.py` — 23 tests

Covers the Horizon 1 and 2 analysis modules.

| Test group | Guards against |
|---|---|
| Evaluation context | An analysis run against a different split than the model was fitted on |
| Fairness | Subgroups not summing to the test set; rates outside [0,1]; a report that omits its decision |
| Model card currency | The card still claiming no fairness audit exists after one has been run |
| Calibration | ECE that fails to detect known over-confidence; bins that lose rows; calibration silently changing the ranking |
| Threshold | Recall rising as the threshold rises; the core economic claim (higher miss cost → lower threshold) failing |
| Drift | PSI non-zero on identical distributions; the detector firing on unshifted data, or failing to fire on shifted data; a baseline built from anything but the training split |

## 6. `tests/test_app_features.py` — 31 tests

Imports from `deploy/` exactly as the running container does, so a broken import path fails
here rather than in the Space.

| Test group | Guards against |
|---|---|
| Explainability | A decomposition that does not reconstruct the model; encoded names leaking into the interface; a missing causal disclaimer |
| Batch validation | Missing columns, negative values or oversized uploads being accepted; unknown categories wrongly blocking |
| Batch scoring | Queue not ranked by risk; batch disagreeing with single-record scoring; risk bands not matching the schema; scoring too slow for the interface |
| Retention brief | The LLM layer being enabled by default; prohibited phrasing passing the filter; the deterministic fallback failing its own checks; raw customer data reaching the prompt |

---

## Deliberately not tested, and why

| Not tested | Reason |
|---|---|
| Exact metric values | Would break on any legitimate model improvement. Metrics are asserted to be numeric and in range; their values are recorded in `model_comparison.csv` and the metadata. |
| Streamlit UI rendering | No browser driver in CI. Covered by the `verify_release.sh` smoke test, which starts the real server and requires HTTP 200 from `/_stcore/health`, and by manual verification in both themes. |
| Docker build in CI | Would add several minutes to every run for a configuration already asserted statically and built locally. The Space build is the real integration test. |
| Live Hugging Face deployment | Requires credentials that must not exist in CI. Verified by observing the platform. |
| Live LLM generation | Calling a real inference provider in CI would need a token in the test environment and would make the suite depend on an external service. The guardrails, the fallback and the disabled-by-default behaviour are all tested; only the network call is not. |

---

## Latest run

```
106 passed
```

Recorded results and timestamps are in `docs/QUALITY_GATE_RESULTS.md`.

## Adding a test

1. Put it in the module matching what it guards.
2. Give it a name stating the property, not the mechanism.
3. Add a row to the table above with the failure mode it prevents.
4. If it needs the model, use the `pipeline` fixture — it skips cleanly when the artifact
   is absent, so a fresh clone does not produce noisy failures before `make train`.
