# California House Price Predictor 🏠

A machine learning project that predicts median house values for California districts using the California Housing dataset from the 1990 U.S. Census.

The project covers data cleaning, feature engineering, exploratory data analysis, model training, evaluation, and a Streamlit prediction app.

## Problem Statement

The goal of this project is to predict the median house value of a California district using information such as location, housing age, number of rooms and bedrooms, population, household size, income, and ocean proximity.

## Dataset

- **Dataset:** California Housing dataset (1990 U.S. Census)
- **Rows:** 20,640
- **Columns:** 10
- **Target:** `median_house_value`

## Features

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity`

## Project Structure

### Data Cleaning

I cleaned the dataset before training the models.

- `total_bedrooms` contained 207 missing values, which were filled using the median grouped by `ocean_proximity`.
- No duplicate rows were found.
- Latitude and longitude values were checked for valid geographic ranges.
- Around 4.68% of the target values were capped at $500,001 in the original dataset.

### Feature Engineering

I created three additional features:

| Feature | Formula |
|---|---|
| `rooms_per_household` | `total_rooms / households` |
| `bedrooms_per_room` | `total_bedrooms / total_rooms` |
| `population_per_household` | `population / households` |

The `ocean_proximity` feature was converted into numerical values using one-hot encoding.

### Exploratory Data Analysis

Some of the main findings from the analysis were:

- `median_income` was the strongest numeric predictor of house value, with a correlation of approximately 0.69.
- Location had a strong effect on house prices.
- Coastal areas generally had higher house values than inland areas.
- NEAR BAY and NEAR OCEAN areas showed higher median values compared with many INLAND areas.
- The distribution of house prices was right-skewed, with many values reaching the $500,001 cap.

### Model Training

I trained and compared several regression models using an 80/20 train-test split.

| Model | RMSE ($) | MAE ($) | R² |
|---|---|---|---|
| Linear Regression | 72,563 | 50,866 | 0.598 |
| Gradient Boosting | 53,510 | 36,486 | 0.781 |
| Random Forest | 50,122 | 32,223 | 0.808 |
| Random Forest (Tuned) | 49,758 | 32,105 | 0.811 |

The tuned Random Forest performed the best, achieving an R² score of 0.811.

I used `RandomizedSearchCV` to tune the Random Forest model. The most important features included:

- `median_income`
- `ocean_proximity_INLAND`
- `population_per_household`

### Streamlit App

I created an interactive Streamlit web application for the trained model.

The app allows users to enter housing information such as:

- Latitude and longitude
- Median income
- Housing age
- Number of rooms
- Number of bedrooms
- Population
- Households
- Ocean proximity

The application then displays the predicted median house value.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- RandomizedSearchCV
- Matplotlib
- Streamlit
- Jupyter Notebook
- Joblib

## Results

The final tuned Random Forest model achieved:

- **R² Score:** 0.811
- **RMSE:** $49,758
- **MAE:** $32,105

This project helped me practice the complete machine learning workflow, from data preprocessing and exploratory analysis to model training, evaluation, and building an interactive application.
