# Security Policy

## Identified Issues and Mitigations

### 1. PII Data Exposure in Notebook Outputs (HIGH)

**Issue:** The Jupyter notebook contains rendered cell outputs with personal financial
data (applicant income, loan amounts, gender, marital status). If the training data
contains real applicant records, this constitutes PII leakage.

**Mitigation:**
- Clear all cell outputs before committing: `jupyter nbconvert --clear-output --inplace *.ipynb`
- The `.gitignore` now prevents raw data files (`.csv`, `.xls`) from being committed.
- Use synthetic or anonymized data for development.

### 2. Unpinned Dependencies (HIGH)

**Issue:** No `requirements.txt` existed, leaving the project vulnerable to supply-chain
attacks (dependency confusion, typosquatting, compromised releases).

**Mitigation:** Added `requirements.txt` with version bounds. For production deployments,
use `pip install --require-hashes` with a lockfile.

### 3. Missing .gitignore (HIGH)

**Issue:** Without a `.gitignore`, sensitive files (`.env`, credentials, data files,
model artifacts containing embedded training data) could be accidentally committed.

**Mitigation:** Added comprehensive `.gitignore` covering secrets, data files, model
artifacts, and environment directories.

### 4. Blanket Warning Suppression (MEDIUM)

**Issue:** `warnings.filterwarnings('ignore')` silences ALL warnings, including those
about deprecated insecure functions and known vulnerabilities.

**Mitigation:** Replaced with targeted suppression of only non-security-relevant
warnings (FutureWarning, seaborn UserWarning).

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by opening
a private issue or contacting the maintainer directly.
