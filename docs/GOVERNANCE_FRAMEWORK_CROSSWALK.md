# Governance Framework Crosswalk

**What this is.** A mapping from three widely-cited AI governance frameworks — the **NIST AI
Risk Management Framework (AI RMF 1.0)**, **ISO/IEC 42001:2023** (AI management systems), and
the **EU Artificial Intelligence Act** (Regulation (EU) 2024/1689) — onto the governance work
this project has already done and published. Nothing in this document adds a new capability;
it takes the fairness audit, calibration analysis, model card, security controls, drift
apparatus and human-review requirement that already exist and shows precisely which
framework clause each one answers.

**What this is not.** This is **not a compliance certification**, **not legal advice**, and
**not a claim of conformity** with any of these three instruments. None of them has been
formally assessed by a qualified auditor or lawyer against this project. It is a **self-issued
crosswalk** — the same kind of artefact a data-science team would prepare *before* commissioning
a formal audit, to show a reviewer where the existing evidence already sits and where it does
not. Every claim below points at a specific file in this repository so it can be checked rather
than taken on trust.

---

## 1. Why this exists

The rest of this project already does the substantive governance work: a fairness audit with
a stated optimisation criterion (`reports/fairness_report.md`), a calibration analysis
(`reports/calibration_report.md`), an exact per-prediction explanation
(`deploy/explain.py`), a drift-detection apparatus (`reports/drift_report.md`), a documented
credential and container-hardening model (`docs/SECURITY.md`), and a standing rule that no
output may drive an automated customer-treatment decision (`deploy/artifacts/model_card.md`).

What was missing was the thing a C-level or model-risk reviewer asks next: *"does any of that
map to something we'd recognise?"* This document is that mapping, so the answer is a table
rather than a re-explanation from first principles.

---

## 2. NIST AI Risk Management Framework (AI RMF 1.0)

The AI RMF organises risk management into four functions — **Govern, Map, Measure, Manage** —
applied iteratively rather than as a linear checklist. It is voluntary and non-prescriptive: it
defines outcomes, not required controls.

| Function | What the function asks | Evidence in this project | File |
|---|---|---|---|
| **Govern** | Is there a risk-aware culture, clear accountability, and are legal/regulatory obligations understood and documented? (This is the substance of subcategory **GOVERN 1.1** specifically.) | The model card states out-of-scope uses explicitly (no autonomous pricing, contract or service decisions); `docs/SECURITY.md` documents the credential and container model; `docs/DECISIONS.md` records 32 numbered design decisions with alternatives considered, which is the accountability trail GOVERN asks for. | `deploy/artifacts/model_card.md`, `docs/SECURITY.md`, `docs/DECISIONS.md` |
| **Map** | Is the system's context, intended purpose and reasonably foreseeable impact identified before deployment? | The model card's "Intended use" and "Out-of-scope uses" sections; the fictional-dataset disclaimer restated in every surface (README, app, Space README, model card) so the operating context is never misstated. | `deploy/artifacts/model_card.md`, `README.md` |
| **Measure** | Are the system's trustworthy characteristics — validity, reliability, safety, fairness, explainability — evaluated and the results documented? Subcategory **MEASURE 2.11** asks specifically that *"fairness and bias... are evaluated and results are documented."* | Fairness audit with three named criteria, a stated optimisation target, and a measured counterfactual cost of removing protected attributes; calibration analysis with Brier score, ECE and (as of the addendum) bootstrap confidence intervals; exact log-odds explanation that reconstructs the model to float precision. | `reports/fairness_report.md`, `reports/calibration_report.md`, `deploy/explain.py` |
| **Manage** | Are identified risks prioritised and treated, and is the system monitored for degradation once deployed? | The cost-ratio threshold sweep turns "should we flag more or fewer accounts" into a documented, parameterised decision; the drift apparatus is the monitoring half, explicitly validated to fire on a shift and stay quiet on a control. | `reports/threshold_analysis.md`, `reports/drift_report.md`, `src/drift.py` |

**Where this project's Measure/Manage coverage is thinner than a production system's would need
to be:** the drift apparatus has never observed real drift (the dataset is one cross-section),
so the Manage function's monitoring half is *validated apparatus*, not *evidence of a working
production monitor*. This is stated in `reports/drift_report.md` itself and repeated here
rather than allowed to sound stronger in this document than it does in the source report.

---

## 3. ISO/IEC 42001:2023 — AI management systems

ISO/IEC 42001 is a certifiable management-system standard (the "42001" numbering follows the
same high-level structure as ISO/IEC 27001 for information security). Clauses 4–10 cover
organisational context, leadership, planning, support, operation, performance evaluation and
improvement; **Annex A** supplies 38 controls grouped into 9 control objectives which an
organisation selects from and records in a **Statement of Applicability (SoA)**.

This project is a single system built by a three-person team for an academic submission, not
an organisation with a certifiable management system — so the mapping below is necessarily
partial, and is presented as such.

| Annex A control objective | What it covers | Evidence in this project | File |
|---|---|---|---|
| **AI system impact assessment** | A documented assessment of the system's likely impact before deployment. | The fairness audit is, in substance, an impact assessment restricted to two protected-adjacent attributes; its own limitations section states explicitly which further impacts (intersectional effects, other attributes) were **not** assessed. | `reports/fairness_report.md` |
| **Data for AI systems** | Data quality, provenance and lifecycle management. | A 30-check validation gate that training refuses to proceed without; a verified Git blob SHA against the official IBM publication on every run. | `src/data_validation.py`, `SOURCE_MANIFEST.json` |
| **AI system life cycle** | Documented development, verification and change process. | The leakage-safe single-split methodology, the pre-declared model-selection rule, and 146 automated tests that gate every change. | `docs/ARCHITECTURE.md`, `src/train.py`, `tests/` |
| **Responsible use of AI systems** | Constraints on what the system may be used for, and human-oversight requirements. | The model card's governance section and the "human review required" badge repeated on every screen of the application. | `deploy/artifacts/model_card.md`, `deploy/app.py` |
| **Third-party and supplier relationships** | Management of externally-supplied components and data. | The dataset's provenance chain (public repository, Apache-2.0 licence, verified blob SHA) and the explicit dependency-cost reasoning recorded for every third-party library considered and rejected (Fairlearn, AIF360, Evidently, XGBoost). | `LICENSE_NOTICE.md`, `docs/DECISIONS.md` |
| **AI policy** / **internal organisation** / **resources** | Organisation-level policy, roles and resourcing. | **Not applicable at this project's scale.** A three-person academic team does not have — and should not fabricate — an organisational AI policy document. Recorded as a gap, not papered over. | — |
| **Information for interested parties** | Disclosures to affected parties and users. | The data-and-use disclaimer shown on every page of the application, and the model card's full disclosure of metrics, limitations and governance constraints. | `deploy/app.py`, `deploy/artifacts/model_card.md` |

**No Statement of Applicability exists**, and none is claimed. The table above is the informal
equivalent — which controls apply and where the evidence lives — without the formal SoA
document ISO/IEC 42001 certification would require.

---

## 4. EU Artificial Intelligence Act (Regulation (EU) 2024/1689)

### 4.1 Does this project's use case actually fall under the Act's high-risk tier?

**This is the question a governance reviewer asks first, and the honest answer is: almost
certainly not, and here is the specific reasoning rather than an assumption.**

High-risk status under the Act is determined by **Annex III**, an exhaustive list of eight use
areas: biometrics; critical infrastructure; education and vocational training; employment and
worker management; **access to essential services** (which explicitly includes credit
scoring and insurance risk/pricing); law enforcement; migration and border control; and
administration of justice.

A telecommunications **retention-prioritisation** tool — deciding which existing customers a
human specialist reviews first — is not a listed use case. It is not credit scoring (it does
not evaluate creditworthiness or gate access to a loan or financial product), it is not an
employment decision, and it does not determine access to an essential service in the sense
Annex III uses the term (it does not decide *whether* a customer may have the service — every
customer already has it — only which existing accounts a human reviews first).

**Consequence.** Article 86 (right to explanation of individual decision-making) and the bulk
of the Chapter III high-risk obligations (Articles 9–15, 72) apply specifically to Annex III
systems. On the reasoning above, they are **very likely not legally triggered** for this
platform's actual use case. **This is not a legal opinion** — an actual deployer processing
real customer data should have this classification confirmed by counsel, not inferred from a
project README. It is stated here so the crosswalk that follows is read for what it is: *"if
this were an in-scope system, here is where the evidence already sits"* — a preparedness
exercise, not a claim of exemption relied upon in production.

### 4.2 The crosswalk, on the assumption the system were in scope

| Article | Requirement | Evidence in this project | File |
|---|---|---|---|
| **Art. 9** — Risk management system | A continuous, documented risk-management process across the system's lifecycle. | The fairness audit, calibration analysis, threshold sensitivity analysis and drift apparatus are, collectively, exactly this — each closes one governance gap the project published about itself before closing it. | `docs/IMPROVEMENT_PLAN.md` §"Gap analysis" |
| **Art. 10** — Data and data governance | Training/validation/test data examined for relevant biases; documented data governance. | The fairness audit's subgroup base-rate analysis is a direct bias examination of the training data, not just the model; the data-validation gate documents provenance and quality. | `reports/fairness_report.md`, `src/data_validation.py` |
| **Art. 12** — Record-keeping | Automatic logging of events across the system's lifetime, tamper-evident and queryable. | **Partial.** MLflow experiment tracking logs training runs with a documented rollback; there is **no request-level inference log** of individual predictions in production — the Streamlit application logs to stdout only, and batch uploads are explicitly never persisted (by design, for a different reason: the batch page promises uploaded customer data is never written to disk). This is a genuine gap against Art. 12 if the system were in scope, not a satisfied requirement. | `reports/tracking_report.md` |
| **Art. 13** — Transparency | Instructions enabling a deployer to interpret the system's output correctly. | The model card's metrics, limitations and selection-rule sections; the in-app caption stating the probability is "a model score, not a validated frequency." | `deploy/artifacts/model_card.md`, `deploy/app.py` |
| **Art. 14** — Human oversight | The system must be designed so a human can understand, monitor and intervene in its output. | The standing rule, stated in the model card, the Space README and on every result screen: every output requires human review before a customer is contacted, and the model has no authority to act. | `deploy/artifacts/model_card.md`, `deploy/app.py` |
| **Art. 15** — Accuracy, robustness, cybersecurity | Appropriate accuracy metrics disclosed; resilience against errors and attacks. | ROC-AUC/recall/precision/F1 disclosed with a held-out evaluation methodology immune to test-set leakage; the calibration analysis directly addresses the "accuracy" half by distinguishing ranking quality from probability accuracy; container hardening (non-root user, pinned dependencies, secret scanning) addresses the cybersecurity half. | `reports/calibration_report.md`, `docs/SECURITY.md` |
| **Art. 72** — Post-market monitoring plan | A documented plan for monitoring the system's performance after deployment. | **Gap, stated plainly.** The drift apparatus is validated tooling, not an operating post-market monitoring plan — there is no real live population to monitor, and `reports/drift_report.md` says so directly. | `reports/drift_report.md` |
| **Art. 86** — Right to explanation | Affected individuals may request an explanation of a decision's role and main elements. | **Exceeds what would be required** even if in scope: the per-prediction contribution breakdown is exact (reconstructs the model's log-odds to floating-point precision), not an approximation, and is shown by default rather than only on request. | `deploy/explain.py` |

---

## 5. What none of the three frameworks' mapping should be read to imply

- **No external audit has occurred.** Every row above is a first-party assertion, checkable
  against a specific file, but not independently verified by a certification body, a regulator,
  or counsel.
- **No claim of NIST, ISO or EU conformity is being made.** Each framework has requirements
  beyond what a three-person academic project can or should attempt — an organisational AI
  policy, a certified management system, a Data Protection Impact Assessment, a notified-body
  conformity assessment — and this document does not pretend those exist.
- **The EU AI Act applicability assessment in §4.1 is this project's own reasoning**, not a
  legal determination. A live deployment on real customer data must have this confirmed
  properly before relying on it.
- **Frameworks change.** The AI Act's high-risk provisions phase in on a staged timeline; NIST
  and ISO both publish periodic updates and profiles. This crosswalk reflects each framework's
  structure as understood at the time it was written and should be re-verified against the
  current text before being cited in a real governance review.

---

## 6. Maintaining this document

This crosswalk should be revisited whenever a governance-relevant artefact changes:

- A new report added under `reports/` that closes or exposes a gap in §§2–4.
- A materially different finding in the fairness audit, calibration analysis or drift report.
- A version change to any of the three frameworks themselves.

It is a mapping onto evidence that already exists elsewhere in this repository — if a linked
report is regenerated (`make analysis`) and its finding changes, the row that cites it should be
re-checked rather than assumed to still be accurate.
