# Online Advertising Performance Analysis and Click Prediction

A Python data science project , using the supplied Company X dataset from 1 April to 30 June 2020. All money amounts are USD.

## Start here

1. Extract the ZIP. In VS Code, choose **File → Open Folder** and select `advertising_project` (the folder containing `analysis.py`).
2. Use Python 3.12. Install the VS Code Python extension. Install the Jupyter extension if you want to use the notebook.
3. Open **Terminal → New Terminal**. On Windows, run these commands one at a time:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analysis.py
```

Using the interpreter path directly avoids PowerShell activation-policy issues. If `py -3.12` is unavailable, install Python 3.12 first, or use `python -m venv .venv` after checking your Python version.

On macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python analysis.py
```

Open `outputs/report.html` in your web browser. The included report already contains the completed results, so you can read it before installing anything. The script regenerates all outputs. Internet access is needed to install dependencies, not to run the analysis.

## Notebook in VS Code

Install the optional kernel package in your environment:

```powershell
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Open `advertising_analysis.ipynb`, choose **Select Kernel → Python Environments → .venv**, and click **Run All**. The notebook uses the same tested functions as the script. Its cells follow the 20 questions and display the result tables and plots. The notebook starts without saved cell outputs; the completed HTML report is included separately.

## Project contents

- `analysis.py`: data validation, cleaning, grouped metrics, all 20 answers, six chart files, chronological ML evaluation, report generation.
- `advertising_analysis.ipynb`: guided notebook for learning and presenting the analysis.
- `data/online_advertising_performance_data.csv`: original dataset, unchanged.
- `outputs/report.html`: self-contained report with all 20 answers, charts, and model results.
- `outputs/*.csv`: cleaned data, grouped results, outlier records, duplicate sensitivity, model scores, and June predictions.
- `outputs/data_quality.json`: data quality audit.
- `requirements.txt`: versions used to run this project.

## Problem statement

Analyze how campaign, banner, placement, audience category, and date relate to advertising performance. Identify descriptive differences in click-through rates, transaction rates, cost efficiency, and return on ad spend. Build and evaluate a model to estimate clicks given an impression volume and campaign configuration.

## Metric definitions — important for your presentation

| Metric | Formula | Interpretation |
|---|---|---|
| CTR (%) | 100 × clicks / displays | Clicks per 100 impressions |
| Post-click transaction rate (%) | 100 × transactions / clicks | Attributed transactions per 100 clicks, not unique-user conversion probability |
| Firm CPC | cost / clicks | Placement cost paid by Advert Firm A per click |
| Company X CPC | revenue / clicks | Company X ad spend per click |
| Firm CPA | cost / transactions | Firm cost per attributed transaction |
| Company X CPA | revenue / transactions | Company X spend per attributed transaction |
| Firm ROI (%) | 100 × (revenue − cost) / cost | Return on placement cost, before other firm expenses |
| Company X ROAS | attributed sales / revenue | Sales dollars per dollar of Company X recorded ad spend |

The column `revenue` is revenue **for Advert Firm A**, and expense **for Company X**. Company X profit ROI cannot be calculated without product costs, margins, and other expenses. Never label `(sales − cost) / cost` as Company X ROI here.

All grouped rates divide total numerators by total denominators. A zero denominator produces an undefined value (NaN), not zero. Audience mix is weighted by impressions. Weekday activity comparisons use daily averages because the number of each weekday differs.

## Cleaning decisions

There are 15,408 records and 91 dates. The year 2020 comes from the supplied description. Two fully empty columns are removed; 413 missing placements become `Unknown`. Five exact duplicate rows are retained because the source has no record ID; a separate sensitivity table shows the totals if they are removed. Outliers remain in the data.

There are 329 rows where transactions exceed clicks, 31 with transactions but no clicks, and 2 with clicks exceeding displays. These need source clarification. Repeated transactions or attribution conventions could contribute, but the data alone cannot explain them. Results should not be presented as user-level conversion probabilities. The dataset contains audience labels, not direct measurements of an individual's engagement change.

## Machine learning design

**Target:** row-level click count. This is regression, not classification. A row summarizes activity; it is not an individual user.

**Features:** campaign, engagement category, banner, placement, log impression volume, banner area, day of week, and elapsed days. Cost, revenue, transactions, and sales are excluded to avoid outcome leakage.

**Use case:** estimate clicks for a known or planned impression volume. The offline test uses realized displays, so performance is conditional on knowing that volume. This model is not an unconditional future traffic forecast.

**Split:** train on April, choose a model on May MAE, refit on April + May, and evaluate once on June. Compare a training-CTR × displays baseline with two Poisson gradient-boosting configurations. Categories are encoded within the training pipeline. Random row splits are avoided because they mix future and past observations.

**Metrics:** MAE, RMSE, R², and total predicted versus actual clicks. The included model reduces June MAE versus the baseline, but overpredicts June total clicks substantially. Do not call it deployment-ready. The series changes over time, and extrapolating tree models beyond the training period is limited. A next study should collect later data, use rolling temporal validation, and check calibration and stable segment support.

## Analysis limitations

- Only one quarter is observed: annual seasonality cannot be established.
- Placement codes do not identify website/app types or publisher names.
- Volume, audience targeting, and campaign allocation confound comparisons. These are associations, not causal effects of banner size or placement.
- Conversions and sales use a 30-day attribution window. Extraction date is unknown; June values may be incomplete. Future conversion-prediction work must use matured labels and an appropriate 30-day training gap.
- No user IDs, product margins, experiment assignment, or organic-sales counterfactual are supplied.
- Rankings are descriptive. They do not establish significance. The combination shortlist requires 1,000 clicks and 30 transactions, an explicit heuristic support rule.
- Very high ROAS needs validation of tracking and attribution, rather than an immediate budget-allocation conclusion.

## Suggested project presentation flow

Explain the business question and metric ownership first. Show cleaning decisions, then trends, banners, placements and campaigns. Present the modeling target, chronological split, baseline and test results. Finish with limitations and the data you would request next. Use `outputs/report.html` for the exact figures and all 20 answers.

## Reproducibility

Executed successfully with Python 3.12 and the versions in requirements.txt. Random seed is 42. The script has no external data calls. Results may vary slightly across numeric-library platforms. The supplied outputs and the Python execution path were checked; the VS Code UI and notebook kernel were not executed in this environment.
