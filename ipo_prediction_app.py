
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="IPO CMP Prediction App", layout="wide")

st.title("📈 IPO CMP Prediction App")
st.write("This Streamlit app predicts the **Current Market Price (CMP)** of IPOs using Linear Regression and Random Forest models.")

# --- Upload Section ---
st.sidebar.header("Upload IPO Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your IPO CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Data Cleaning ---
    df['Issue_Size(crores)'] = df['Issue_Size(crores)'].astype(str).str.replace(',', '', regex=False)
    df['Issue_Size(crores)'] = pd.to_numeric(df['Issue_Size(crores)'], errors='coerce')

    numeric_cols = [
        "Issue_Size(crores)", "QIB", "HNI", "RII",
        "Listing_Open", "Listing_Close", "Listing_Gains(%)",
        "CMP", "Current_gains"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Feature Engineering
    df_encoded = pd.get_dummies(df, columns=['IPO_Name'], drop_first=True)
    df_encoded['Date'] = pd.to_datetime(df_encoded['Date'], format='%d-%m-%Y', errors='coerce')
    df_encoded['Year'] = df_encoded['Date'].dt.year
    df_encoded['Month'] = df_encoded['Date'].dt.month
    df_encoded.drop(columns=['Date'], inplace=True)

    # Split X and y
    y = df_encoded['CMP']
    X = df_encoded.drop(columns=['CMP'])

    # Standardization
    scaler = StandardScaler()
    numeric_cols_for_scaling = [c for c in numeric_cols if c in X.columns]
    numeric_cols_for_scaling += ['Year', 'Month']
    X[numeric_cols_for_scaling] = scaler.fit_transform(X[numeric_cols_for_scaling])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Linear Regression ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # --- Random Forest ---
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # --- Evaluation ---
    metrics = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest'],
        'MAE': [
            mean_absolute_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_rf)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            np.sqrt(mean_squared_error(y_test, y_pred_rf))
        ],
        'R²': [
            r2_score(y_test, y_pred_lr),
            r2_score(y_test, y_pred_rf)
        ]
    })

    st.subheader("📊 Model Performance Comparison")
    st.dataframe(metrics)

    st.bar_chart(metrics.set_index("Model")["R²"])

    # --- Prediction Section ---
    st.sidebar.header("Predict New IPO CMP")

    issue_size = st.sidebar.number_input("Issue Size (crores)", min_value=1.0, value=1300.0)
    qib = st.sidebar.slider("QIB (%)", 0, 100, 75)
    hni = st.sidebar.slider("HNI (%)", 0, 100, 15)
    rii = st.sidebar.slider("RII (%)", 0, 100, 10)
    issue_price = st.sidebar.number_input("Issue Price (₹)", min_value=1.0, value=675.0)
    listing_open = st.sidebar.number_input("Listing Open (₹)", min_value=1.0, value=1015.0)
    listing_close = st.sidebar.number_input("Listing Close (₹)", min_value=1.0, value=1015.0)
    listing_gains = st.sidebar.number_input("Listing Gains (%)", value=50.37)
    current_gains = st.sidebar.number_input("Current Gains (%)", value=64.81)
    year = st.sidebar.number_input("Year", value=2025)
    month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=10)

    # Prepare test input
    test_data = pd.DataFrame({
        'Issue_Size(crores)': [issue_size],
        'QIB': [qib],
        'HNI': [hni],
        'RII': [rii],
        'Issue_price': [issue_price],
        'Listing_Open': [listing_open],
        'Listing_Close': [listing_close],
        'Listing_Gains(%)': [listing_gains],
        'Current_gains': [current_gains],
        'Year': [year],
        'Month': [month]
    })

    # Ensure same feature structure
    for col in X.columns:
        if col not in test_data.columns:
            test_data[col] = 0
    test_data = test_data[X.columns]

    test_data[numeric_cols_for_scaling] = scaler.transform(test_data[numeric_cols_for_scaling])

    if st.sidebar.button("Predict CMP"):
        predicted_cmp = rf_model.predict(test_data)[0]
        st.success(f"💰 Predicted CMP: ₹{predicted_cmp:.2f}")

        # Optional: comparison plot
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Predicted CMP'], [predicted_cmp], color='lightgreen')
        ax.set_ylabel('CMP (₹)')
        ax.set_title('Predicted CMP Value')
        st.pyplot(fig)

else:
    st.info("👈 Please upload your IPO dataset to start the analysis.")
