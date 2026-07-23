# Exploratory Data Analysis — Observations

Every figure in this document is produced by `src/eda.py` from the validated raw
file `data/raw/Telco-Customer-Churn.csv` (Git blob SHA
`3de7a612d1609f25f21a455bda77948729369002`). No value here was entered by hand.

The dataset is IBM's **fictional** telecommunications sample. Observations below
describe that sample, and associations described are not evidence of causation.

## 1. Target balance

- 7,043 customers; 1,869 churned and 5,174 did not.
- The churn rate in the sample is **26.54%**, so the classes are imbalanced
  at roughly 1 churner to 2.8 non-churners.
- Consequence for modelling: a model that predicts "no churn" for every customer would
  reach 73.46% accuracy while identifying no churn risk at all. Accuracy
  alone is therefore not an acceptable selection metric; recall, F1 and ROC-AUC are used.

## 2. Data quality

- After stripping whitespace and coercing `TotalCharges` to numeric, 11 values
  are missing in the whole table, all of them in `TotalCharges` (11 rows).
- Those 11 rows all have `tenure = 0`, i.e. customers who have not yet
  completed a billing cycle. The blank is structurally meaningful rather than random.
- The raw file is left untouched. Missing values are handled by a median imputer that is
  fitted **inside the pipeline on the training split only**, so no test information leaks.
- No duplicate rows and no duplicate `customerID` values were found.

## 3. Tenure

- Mean tenure of churned customers is 17.98 months against
  37.57 months for retained customers.
- Median tenure is 10.0 months for churners and
  38.0 months for non-churners.
- Churn is concentrated among recently acquired customers in this sample. Early-life
  accounts may therefore warrant closer retention attention.

## 4. Charges

- Mean monthly charge is 74.44 for churners and
  61.27 for retained customers.
- Mean total charge is 1531.80 for churners and
  2555.34 for retained customers, which is consistent with
  churners having shorter tenure rather than with lower spending per month.

## 5. Contract term

| Contract | Customers | Churn rate |
|---|---:|---:|
| Month-to-month | 3,875 | 42.71% |
| One year | 1,473 | 11.27% |
| Two year | 1,695 | 2.83% |

- `Month-to-month` contracts show the highest churn rate in the sample
  (42.71%), against
  2.83% for `Two year`.
- Contract term is the single strongest categorical separator observed. This is an
  association within the sample, not proof that changing a contract changes behaviour.

## 6. Internet service and add-ons

| Internet service | Customers | Churn rate |
|---|---:|---:|
| Fiber optic | 3,096 | 41.89% |
| DSL | 2,421 | 18.96% |
| No | 1,526 | 7.40% |

- `Fiber optic` customers churn at 41.89%,
  compared with 7.40% for `No`.
- Customers without technical support churn at 41.64% versus 15.17% for those with it.
- Customers without online security churn at 41.77% versus 14.61% for those with it.
- Note the confound: the 'No internet service' level appears in every add-on column, so
  these add-on comparisons partly restate the internet-service split.

## 7. Payment method

| Payment method | Customers | Churn rate |
|---|---:|---:|
| Electronic check | 2,365 | 45.29% |
| Mailed check | 1,612 | 19.11% |
| Bank transfer (automatic) | 1,544 | 16.71% |
| Credit card (automatic) | 1,522 | 15.24% |

- `Electronic check` shows the highest churn rate at
  45.29%.

## 8. Numeric correlations

- `tenure` correlates with the churn flag at -0.352.
- `MonthlyCharges` correlates at 0.193.
- `TotalCharges` correlates at -0.200.
- `tenure` and `TotalCharges` are strongly related (0.826), which is expected because total charges
  accumulate with time. Tree ensembles tolerate this; the linear model uses scaling and
  regularisation, and the correlation is recorded here as a known limitation.

## 9. Implications carried into modelling

1. Class imbalance means recall and ROC-AUC drive model selection, not accuracy.
2. `TotalCharges` needs coercion and imputation inside the pipeline, never beforehand.
3. `SeniorCitizen` is treated as a binary category, not a continuous magnitude.
4. `customerID` is excluded from the feature set entirely.
5. Contract, internet service, tenure and payment method are the most promising signals
   and are all retained as predictors.

## Figures

All figures are saved to `reports/figures/` at 200 dpi:

| File | Content |
|---|---|
| `01_target_distribution.png` | Target class balance |
| `02_missing_values.png` | Missing values after coercion |
| `03_tenure_by_churn.png` | Tenure distribution by churn |
| `04_monthlycharges_by_churn.png` | Monthly charges distribution by churn |
| `05_totalcharges_by_churn.png` | Total charges distribution by churn |
| `06_churn_rate_by_contract.png` | Churn rate by contract term |
| `07_churn_rate_by_internetservice.png` | Churn rate by internet service |
| `08_churn_rate_by_paymentmethod.png` | Churn rate by payment method |
| `09_churn_rate_by_techsupport.png` | Churn rate by technical support |
| `10_churn_rate_by_onlinesecurity.png` | Churn rate by online security |
| `11_correlation_matrix.png` | Numeric correlation matrix |
