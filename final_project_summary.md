# Final Project Summary

## Business Problem

Small and mid-sized health insurers, Medicaid administrators, and third-party claims processors often rely on manual review and static rules to identify suspicious claims. That approach misses some fraud, creates too many false positives, slows investigations, and increases financial losses. This project builds an AI-assisted fraud detection system that scores providers and claims for fraud risk and supports human investigators with clearer prioritization.

## Datasets Used

### Primary Dataset

`Healthcare Provider Fraud Detection Analysis` from Kaggle is the main modeling dataset. It combines:

- beneficiary-level demographic and chronic condition data
- inpatient claim data
- outpatient claim data
- provider-level fraud labels

The project merges these files and engineers provider-level features so the model predicts whether a provider is potentially fraudulent.

### Secondary Dataset

`Health Insurance Claims Data for Fraud Detection` is a synthetic claim-level dataset with fraud labels. It is used as a comparison model and as a cleaner manual-input demo inside Streamlit.

## Modeling Approach

The primary workflow:

1. load beneficiary, inpatient, outpatient, and provider label files
2. combine inpatient and outpatient claims
3. merge claims with beneficiary records by `BeneID`
4. merge provider fraud labels by `Provider`
5. convert `PotentialFraud` to a binary target
6. aggregate claims into provider-level features

Example provider-level features:

- total claims
- inpatient and outpatient claim counts
- average and total reimbursed claim amount
- average and total deductible amount
- unique beneficiaries
- unique physicians
- average claim duration
- average chronic condition count
- average patient age
- percentage of claims with an admitted diagnosis code
- unique diagnosis and procedure codes

The project compares Logistic Regression, Random Forest, and XGBoost if available. If XGBoost is not installed, the code automatically falls back to sklearn boosting models.

## Evaluation Metrics

Because healthcare fraud detection is imbalanced, the project does not rely only on accuracy. It reports:

- accuracy
- precision
- recall
- F1 score
- ROC-AUC
- confusion matrix
- classification report

It also evaluates business thresholds at `0.30`, `0.50`, and `0.70`.

The risk policy is:

- `0.00-0.30`: Low Risk / Auto Approve
- `0.30-0.70`: Medium Risk / Human Review
- `0.70-1.00`: High Risk / Investigate Before Payment

For this business case, the preferred threshold should balance recall and precision while leaning toward catching more fraud. That is why the project emphasizes recall, precision, F1, ROC-AUC, and top-flagged precision instead of accuracy alone.

## Business Impact

This system helps insurers and claims teams:

- prioritize the highest-risk providers and claims first
- reduce wasted investigator time on low-risk cases
- surface patterns that static rules may miss
- make fraud review more consistent and scalable

Instead of replacing investigators, the model improves triage and speeds up decisions on where human review should focus.

## Responsible AI Controls

- False positives can delay legitimate claims and frustrate providers or patients.
- False negatives can allow fraudulent claims to be paid.
- Claims above `$10,000` that are flagged as suspicious should require human review before denial.
- Predictions with model confidence below `80%` should require human verification.
- Providers with no prior fraud history should not be denied automatically.
- The model should assist investigators, not replace them.

## Limitations

- The main Kaggle labels reflect potential fraud, not final legal determinations.
- Provider-level prediction may not capture every claim-specific fraud pattern.
- The synthetic dataset is useful for demonstration, but it does not represent real operational complexity.
- Real deployment would need monitoring, drift detection, audit logging, and policy oversight.

## Future Improvements

- add SHAP-based local explanations for investigators
- include network relationships across providers, physicians, and beneficiaries
- calibrate probabilities and optimize thresholds against investigation cost
- integrate case management workflow and analyst feedback loops
- retrain periodically with newer claims and confirmed outcomes
