# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TDW6323 Data Wrangling and Visualisation — Lab Assignment (20%)**
Multimedia University group assignment. Work in groups of 3–4 students. All work is done in a single Jupyter Notebook derived from the provided template.

- **Dataset:** `dataset/diabetic_data.csv` (Diabetic_Data.csv)
- **Template:** `template/Lab Assignment_Part1_2_3 Template _updated.ipynb`
- **Submission deadline:** 15 June 2026, 10:00 AM (via eBwise as a single zip file)
- **Late penalty:** 5% deducted per overdue day; resubmission after deadline is not permitted

---

## Environment Setup

This project uses Python with Jupyter Notebook. Required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

Launch Jupyter:
```bash
jupyter notebook
```

Run a specific notebook non-interactively (for validation):
```bash
jupyter nbconvert --to notebook --execute "template/Lab Assignment_Part1_2_3 Template _updated.ipynb"
```

---

## Assignment Structure

The notebook is divided into three parts. All parts use `Diabetic_Data.csv`. Parts 2 and 3 use the **cleaned dataset** produced at the end of Part 1.

### Part 1 — Data Cleaning and Transformation (10 marks)

#### 1.1 Missing Data Handling (4 marks)
- Display and describe the dataset (`df.describe()`, `df.info()`)
- Check shape (rows × cols)
- Transform `readmitted` column and replace `?` / sentinel values with `NaN`
- Identify columns with missing values; choose and justify an imputation strategy
- Implement the imputation; drop non-impactful columns
- Split numerical and categorical (non-numerical) attributes
- Confirm all missing values are resolved

#### 1.2 Data Cleaning and Transformation (3 marks)
- Convert `readmitted` to binary numerical (readmitted = 1, not readmitted = 0); plot class distribution
- Plot impact of `gender`, `age`, and `race` on `readmitted` (histogram or bar graph); explain frequency distribution
- Calculate average insulin rate per readmitted instance, grouped by `('Down', 'No', 'Steady', 'Up')`; visualise insulin impact on readmission by age group, gender, and race
- Clean `diag_1`, `diag_2`, `diag_3` using `fillna` and map numeric ranges to disease categories:

| Range / Value | Category |
|---|---|
| 390–459 or 785 | Circulatory |
| 460–519 or 786 | Respiratory |
| 520–579 or 787 | Digestive |
| 250 | Diabetes |
| 800–999 | Injury |
| 710–739 | Musculoskeletal |
| 580–629 or 788 | Genitourinary |
| 140–239 | Neoplasms |
| -1 | NAN |
| (anything else) | Other |

  Plot frequency distribution by category (bar graph); identify which category is most readmitted.

- Select 21 drug feature columns and map `'No'→0`, `'Steady'→1`, `'Up'→1`, `'Down'→1`; cast columns to `int`
- Transform `A1Cresult`: `'>7'→1`, `'>8'→1`, `'Norm'→0`, `'None'→-99`
- Transform `max_glu_serum`: `'>200'→1`, `'>300'→1`, `'Norm'→0`, `'None'→-99`

#### 1.3 Outliers Detection and Treatment (3 marks)
- Explore potential outliers visually (box plot, scatter plot, etc.)
- Apply exactly **one** detection method: DBSCAN, IQR, or Z-Score
- Identify and visualise outliers per column
- Apply an outlier treatment strategy
- Show **Before & After** comparison

---

### Part 2 — Feature Selection (5 marks)

Operates on the cleaned dataset from Part 1.

- **Correlation heatmap:** filter to numerical columns only; plot heatmap; explain which features to select
- **Forward selection (filter method):** use linear regression with `k_features=5`; plot selected features as a bar graph (use `mlxtend.feature_selection.SequentialFeatureSelector`)
- **Lasso Regression (embedded method):** fit Lasso; plot non-zero coefficient features as a bar graph

---

### Part 3 — Data Visualisation (5 marks)

Operates on the cleaned dataset from Part 1.

- Select any **three numerical variables**
- For each variable, produce using Pandas/Matplotlib/Seaborn: histogram, bar graph, pie chart, heatmap
- Briefly explain each chart

---

## Notebook Conventions

- The submitted notebook **must run end-to-end without errors** (`Kernel → Restart & Run All` before submission)
- Keep the section headers from the template intact — they map directly to grading criteria
- Plots must render inline and be visible in the saved notebook
- Add markdown cells below each code cell to explain charts and justify decisions
- Save the completed notebook as `Lab Assignment_Part1_2_3 [Group_Number].ipynb`

---

## Submission Checklist

Zip file contents submitted to eBwise:
- [ ] `Lab Assignment_Part1_2_3 [Group_Number].ipynb` — error-free, all outputs visible
- [ ] Completed Group Assignment Declaration Form (missing form = mark deduction)

Plagiarism: direct copying of text or figures without citation results in **5 marks deducted**.
