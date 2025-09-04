import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # ✅ ADD THIS
import matplotlib.pyplot as plt

# Load trained model and feature columns
model = pickle.load(open("xgb_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


st.title("🚗 Car Price Predictor")

# --- User Input ---
year = st.number_input("Year of Purchase", 2000, 2023, step=1)
kms = st.number_input("Kilometers Driven", 0, 1000000, step=100)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
seller = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
brand = st.selectbox("Car Brand", [col.replace("brand_", "") for col in columns if col.startswith("brand_")])

# --- Create input vector ---
input_dict = dict.fromkeys(columns, 0)
input_dict['year'] = year
input_dict['km_driven'] = kms

# Set one-hot fields
brand_col = f"brand_{brand}"
if brand_col in input_dict:
    input_dict[brand_col] = 1

fuel_col = f"fuel_{fuel}"
if fuel_col in input_dict:
    input_dict[fuel_col] = 1

seller_col = f"seller_type_{seller}"
if seller_col in input_dict:
    input_dict[seller_col] = 1

trans_col = f"transmission_{transmission}"
if trans_col in input_dict:
    input_dict[trans_col] = 1

owner_map = {
    0: 'First Owner',
    1: 'Second Owner',
    2: 'Third Owner',
    3: 'Fourth & Above Owner'
}
owner_label = owner_map.get(owner, 'First Owner')
owner_col = f'owner_{owner_label}'
if owner_col in input_dict:
    input_dict[owner_col] = 1

input_df = pd.DataFrame([input_dict])

# --- Predict ---
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    price_in_lakhs = prediction / 100000
    st.success(f"Estimated Selling Price: ₹ {price_in_lakhs:,.2f} lakhs")

    # Show SHAP Explanation if enabled
    if st.checkbox("🔍 Show SHAP Explanation"):
        st.subheader("Why this prediction?")
        st.write("✅ SHAP block running...")  # Debug marker

        # Create SHAP explainer and values
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # Plot SHAP using matplotlib
        fig, ax = plt.subplots(figsize=(10, 5))  # ✅ Create a figure
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)  # ✅ Show it in Streamlit
        plt.clf()       # ✅ Clear figure after showing
