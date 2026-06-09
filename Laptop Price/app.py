import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load Model
# -------------------------

model = joblib.load(
    "models/laptop_price_model.pkl"
)

# -------------------------
# UI
# -------------------------

st.title("💻 Laptop Price Predictor")

st.write(
    "Predict laptop price using Machine Learning"
)

# -------------------------
# Inputs
# -------------------------

brand = st.text_input(
    "Brand",
    "HP"
)

name = st.text_input(
    "Laptop Name",
    "15s"
)

processor = st.text_input(
    "Processor",
    "12th Gen Intel Core i5 1235U"
)

cpu = st.text_input(
    "CPU Details",
    "10 Cores, 12 Threads"
)

ram = st.slider(
    "RAM (GB)",
    4,
    64,
    8
)

ram_type = st.selectbox(
    "RAM Type",
    [
        "DDR4",
        "DDR5",
        "LPDDR4",
        "LPDDR4X",
        "LPDDR5",
        "LPDDR5X",
        "Unified"
    ]
)

spec_rating = st.slider(
    "Specification Rating",
    50,
    100,
    75
)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Price"):

    sample = pd.DataFrame({
        "brand":[brand],
        "name":[name],
        "spec_rating":[spec_rating],
        "processor":[processor],
        "CPU":[cpu],
        "Ram":[ram],
        "Ram_type":[ram_type]
    })

    prediction = model.predict(sample)[0]

    st.success(
        f"Estimated Price: ₹ {prediction:,.0f}"
    )