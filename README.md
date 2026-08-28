# California House Price Predictor 🏠

Predicting median house values for California districts using 1990 census
block-group data — a full pipeline from raw data to a deployed prediction app.

## Problem Statement

Given demographic and geographic statistics for a California census block
group (location, housing age, room/bedroom counts, population, income, and
proximity to the ocean), predict the **median house value** for that
district.

## Dataset

- **Source:** California Housing dataset (1990 U.S. Census), 20,640 rows / 10 columns.
- **Target:** `median_house_value`
- **Features:** `longitude`, `latitude`, `housing_median_age`, `total_rooms`,
  `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`

## Project Structure

```
housing_project/
├── data/
│   ├── housing.csv              # raw dataset
│   ├── housing_clean.csv        # after cleaning
│   └── housing_features.csv     # after feature engineering
├── figures/                      # saved EDA + feature importance plots
├── models/
│   ├── best_model.pkl             # final trained model (compressed, ~23MB)
│   ├── feature_columns.pkl        # exact feature order used at train time
│   └── results.json               # model comparison metrics
├── notebooks/
│   ├── housing_analysis.ipynb     # single narrated notebook (cleaning → EDA → modeling)
│   └── pipeline/                  # the 4 pipeline stages as standalone notebooks
│       ├── 01_clean_data.ipynb
│       ├── 02_feature_engineering.ipynb
│       ├── 03_eda.ipynb
│       └── 04_train_model.ipynb
├── app.py                        # Streamlit prediction app
└── requirements.txt
```

Each notebook in `notebooks/pipeline/` auto-detects the project root on its
first cell, so it works regardless of where Jupyter was launched from.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/pipeline/
# run 01_clean_data.ipynb, then 02_feature_engineering.ipynb,
# then 03_eda.ipynb, then 04_train_model.ipynb, in that order
streamlit run app.py
```

## 1. Data Cleaning

- `total_bedrooms` had 207 missing values (~1%) — imputed with the **median,
  grouped by `ocean_proximity`**, so districts of similar type get a more
  realistic fill value than a single global median.
- No duplicate rows found.
- Sanity-checked that all latitude/longitude values fall within California's
  geographic bounds.
- Flagged that **4.68% of rows have `median_house_value` capped at $500,001**
  — this is a known artifact of the original census data collection (values
  were top-coded), not a real price ceiling. It's kept in the training data
  by default but the cleaning script supports dropping it via
  `clean_data(df, drop_capped_target=True)`.

## 2. Feature Engineering

Three engineered ratio features consistently added the most predictive value
beyond the raw counts:

| Feature | Formula |
|---|---|
| `rooms_per_household` | `total_rooms / households` |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` |
| `population_per_household` | `population / households` |

`ocean_proximity` was one-hot encoded into 5 binary columns. A log-transformed
target (`log_median_house_value`) was also created to reduce right-skew, for
use with linear models if desired.

## 3. EDA — Key Findings

*(see `figures/` for the full set of saved plots)*

- **Median income is by far the strongest predictor** (correlation ≈ 0.69
  with house value) — no other numeric feature comes close.
- **Geography matters**: plotting longitude/latitude colored by house value
  reproduces the shape of California and shows a clear coastal price premium,
  especially around the Bay Area and Southern California coast.
- **Ocean proximity**: `ISLAND` and `NEAR BAY`/`NEAR OCEAN` districts have
  noticeably higher median values than `INLAND` districts.
- The target's right-skew and the $500,001 cap are visible directly in the
  histogram of `median_house_value`.

## 4. Modeling

Three baseline models were trained and compared on an 80/20 train/test split:

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression | 72,563 | 50,866 | 0.598 |
| Gradient Boosting | 53,510 | 36,486 | 0.781 |
| Random Forest | 50,122 | 32,223 | 0.808 |
| **Random Forest (tuned)** | **49,758** | **32,105** | **0.811** |

Random Forest was selected as the best baseline and tuned with
`RandomizedSearchCV` (searched on a training subsample for speed, then
refit on the full training set). **`median_income`, `ocean_proximity_INLAND`,
and `population_per_household`** were the top three features by importance.

> **Note on model size:** an uncapped Random Forest (`max_depth=None`,
> `min_samples_leaf=1`) fully memorizes the training data into very deep
> trees, producing a 200MB+ pickle file — too large for a normal GitHub push.
> The tuning search caps `max_depth` and requires `min_samples_leaf >= 2`,
> and the model is saved with `joblib` compression, bringing the final file
> down to ~23MB with virtually no loss in accuracy.

> If XGBoost is installed (`pip install xgboost`), `train_model.py` will
> automatically include it in the comparison — it was excluded from this run
> because it wasn't available in the environment used to build this project,
> but the fallback to `GradientBoostingRegressor` keeps the pipeline fully
> functional either way.

## 5. Frontend

A **Streamlit app** (`app.py`) provides an interactive form:
- Sliders for latitude/longitude, income, and housing age
- A dropdown for ocean proximity
- Number inputs for rooms, bedrooms, population, and households
- Outputs the predicted median house value plus a small map pin of the
  entered location

Deploy for free on **Streamlit Community Cloud** by connecting this GitHub
repo — the app will auto-build from `requirements.txt` and `app.py`.

## 6. Documentation Notes

- Each pipeline stage (`notebooks/pipeline/*.ipynb`) is a standalone, runnable
  notebook with markdown explanations alongside the code, so the whole
  pipeline is reproducible and readable step-by-step.
- `notebooks/housing_analysis.ipynb` mirrors the same pipeline in a single
  narrated notebook for anyone who wants to read the analysis top-to-bottom
  without running scripts.
- `models/results.json` keeps a permanent record of every model's metrics,
  so future changes can be benchmarked against this baseline.

## Is This a Good GitHub Project?

**Yes, with a caveat.** The California Housing dataset is one of the most
common "starter" datasets in ML (it ships inside `sklearn.datasets`), so it
won't impress on novelty alone. What makes this version stand out:

- Ratio-feature engineering beyond the raw columns
- A documented model comparison (not just "I used Random Forest")
- A **deployed, interactive app** — most tutorial versions of this project
  stop at the notebook, so having a working Streamlit demo is the biggest
  differentiator
- Clear, written EDA insights rather than just plots with no interpretation

**To strengthen it further:**
- Deploy the Streamlit app and put the live link at the top of this README
- Add 2–3 of the actual figures from `figures/` inline in this README
- Pair it on your GitHub profile with a more original project (e.g. your AI
  Tools Recommender System) so your profile shows both "solid fundamentals"
  and "original work"
