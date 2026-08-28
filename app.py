"""
Streamlit app for California House Price Prediction.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")

MODEL_PATH = "models/best_model.pkl"
COLUMNS_PATH = "models/feature_columns.pkl"

OCEAN_PROXIMITY_OPTIONS = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(COLUMNS_PATH)
    return model, feature_columns


def build_input_row(inputs, feature_columns):
    """Turn the raw user inputs into a single-row DataFrame matching training feature order."""
    total_rooms = inputs["total_rooms"]
    households = inputs["households"]
    total_bedrooms = inputs["total_bedrooms"]
    population = inputs["population"]

    row = {
        "longitude": inputs["longitude"],
        "latitude": inputs["latitude"],
        "housing_median_age": inputs["housing_median_age"],
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": inputs["median_income"],
        "rooms_per_household": total_rooms / households,
        "bedrooms_per_room": total_bedrooms / total_rooms,
        "population_per_household": population / households,
    }

    for opt in OCEAN_PROXIMITY_OPTIONS:
        row[f"ocean_proximity_{opt}"] = 1 if inputs["ocean_proximity"] == opt else 0

    df_row = pd.DataFrame([row])
    df_row = df_row[feature_columns]  # enforce exact training column order
    return df_row


def main():
    st.title("🏠 California House Price Predictor")
    st.write(
        "Estimate the median house value for a California district based on "
        "census-style block-group statistics (the same features used in the "
        "classic California Housing dataset)."
    )

    if not (os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH)):
        st.error(
            "Model files not found. Run `python src/clean_data.py`, "
            "`python src/feature_engineering.py`, then `python src/train_model.py` first."
        )
        return

    model, feature_columns = load_model()

    st.subheader("District Location")
    col1, col2 = st.columns(2)
    with col1:
        latitude = st.slider("Latitude", 32.5, 42.0, 34.2, 0.01)
    with col2:
        longitude = st.slider("Longitude", -124.5, -114.0, -118.4, 0.01)

    ocean_proximity = st.selectbox("Ocean Proximity", OCEAN_PROXIMITY_OPTIONS, index=0)

    st.subheader("Housing & Population Stats")
    housing_median_age = st.slider("Median Age of Houses (years)", 1, 52, 25)
    median_income = st.slider("Median Income (in $10,000s, e.g. 5.0 = $50,000)", 0.5, 15.0, 4.0, 0.1)

    col3, col4 = st.columns(2)
    with col3:
        households = st.number_input("Number of Households in District", min_value=1, value=500)
        total_rooms = st.number_input("Total Rooms in District", min_value=1, value=2500)
    with col4:
        population = st.number_input("Total Population in District", min_value=1, value=1200)
        total_bedrooms = st.number_input("Total Bedrooms in District", min_value=1, value=500)

    if st.button("Predict House Value", type="primary"):
        inputs = {
            "longitude": longitude,
            "latitude": latitude,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "median_income": median_income,
            "ocean_proximity": ocean_proximity,
        }

        X_input = build_input_row(inputs, feature_columns)
        prediction = model.predict(X_input)[0]

        st.success(f"### Estimated Median House Value: ${prediction:,.0f}")

        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}), zoom=5)

        with st.expander("See engineered features used by the model"):
            st.dataframe(X_input.T.rename(columns={0: "value"}))

    st.markdown("---")
    st.caption(
        "Model: trained on the California Housing dataset (20,640 block groups, 1990 census). "
        "For a portfolio project — not intended for real-world real estate valuation."
    )


if __name__ == "__main__":
    main()
