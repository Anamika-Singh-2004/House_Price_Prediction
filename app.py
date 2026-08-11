import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, MODEL_PATH, METRICS_PATH, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import clean_data
from src.train import train_model

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.5rem;}
.hero {
    padding: 1.5rem 2rem; border-radius: 18px; color: white;
    background: linear-gradient(135deg,#16324f,#2d607c);
    margin-bottom: 1rem;
}
.hero h1 {margin:0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🏠 House Price Prediction System</h1>
<p>Kaggle housing data • EDA • Feature Engineering • Regression • Evaluation • Prediction</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return clean_data(pd.read_csv(DATA_PATH))

df = load_data()

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Prediction", "📊 EDA", "🤖 Train & Evaluate", "📋 Dataset", "ℹ️ About"]
)

if page == "🏠 Prediction":
    st.subheader("🏠 Predict House Price")

    if not MODEL_PATH.exists():
        st.warning("Model not found. Go to 'Train & Evaluate' and train the model first.")
        st.stop()

    model = joblib.load(MODEL_PATH)

    st.markdown("### Property Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.25)
        sqft_living = st.number_input("Living Area (sqft)", 300, 15000, 2000, 50)
        sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000, 5000, 100)

    with c2:
        floors = st.number_input("Floors", 1.0, 5.0, 1.0, 0.5)
        waterfront = st.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x else "No")
        view = st.slider("View Rating", 0, 4, 0)
        condition = st.slider("Condition", 1, 5, 3)

    with c3:
        sqft_above = st.number_input("Above Ground Area (sqft)", 300, 15000, 1800, 50)
        sqft_basement = st.number_input("Basement Area (sqft)", 0, 6000, 0, 50)
        yr_built = st.number_input("Year Built", 1850, 2026, 2005)
        yr_renovated = st.number_input("Year Renovated (0 if never)", 0, 2026, 0)

    c4, c5 = st.columns(2)
    cities = sorted(df["city"].dropna().astype(str).unique())
    statezips = sorted(df["statezip"].dropna().astype(str).unique())

    with c4:
        city = st.selectbox("City", cities)
    with c5:
        statezip = st.selectbox("State / ZIP", statezips)

    if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
        sample = pd.DataFrame([{
            "bedrooms": bedrooms, "bathrooms": bathrooms,
            "sqft_living": sqft_living, "sqft_lot": sqft_lot,
            "floors": floors, "waterfront": waterfront, "view": view,
            "condition": condition, "sqft_above": sqft_above,
            "sqft_basement": sqft_basement, "yr_built": yr_built,
            "yr_renovated": yr_renovated, "city": city, "statezip": statezip
        }])

        prediction = float(model.predict(sample)[0])
        st.success(f"Estimated House Price: ${prediction:,.0f}")

    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        a,b,c = st.columns(3)
        a.metric("R² Score", f"{m['r2']:.3f}")
        b.metric("MAE", f"${m['mae']:,.0f}")
        c.metric("RMSE", f"${m['rmse']:,.0f}")

elif page == "📊 EDA":
    st.subheader("📊 Exploratory Data Analysis")

    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Features", len(df.columns)-1)
    c.metric("Missing Values", int(df.isna().sum().sum()))
    d.metric("Average Price", f"${df['price'].mean():,.0f}")

    fig = px.histogram(df, x="price", nbins=50, title="House Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df, x="sqft_living", y="price", color="condition",
        hover_data=["bedrooms", "bathrooms", "city"],
        title="Living Area vs Price"
    )
    st.plotly_chart(fig, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="bedrooms", y="price", title="Price by Bedrooms")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_cities = df.groupby("city")["price"].mean().sort_values(ascending=False).head(15).reset_index()
        fig = px.bar(top_cities, x="city", y="price", title="Average Price - Top Cities")
        st.plotly_chart(fig, use_container_width=True)

    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Train & Evaluate":
    st.subheader("🤖 Train & Evaluate Regression Models")

    model_name = st.selectbox("Choose Model", [
        "Random Forest", "Gradient Boosting", "Linear Regression"
    ])
    test_size = st.slider("Test Set Size", 0.10, 0.40, 0.20, 0.05)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training..."):
            _, metrics, y_test, pred = train_model(model_name, test_size)

        st.success("Model trained and saved successfully.")

        a,b,c = st.columns(3)
        a.metric("R²", f"{metrics['r2']:.4f}")
        b.metric("MAE", f"${metrics['mae']:,.0f}")
        c.metric("RMSE", f"${metrics['rmse']:,.0f}")

        results = pd.DataFrame({"Actual": y_test, "Predicted": pred})
        fig = px.scatter(results, x="Actual", y="Predicted",
                         title="Actual vs Predicted Prices", trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

        results["Residual"] = results["Actual"] - results["Predicted"]
        fig = px.histogram(results, x="Residual", nbins=40,
                           title="Residual Distribution")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📋 Dataset":
    st.subheader("📋 Dataset")

    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### Dataset Information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique()) for c in df.columns]
    })
    st.dataframe(info, use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset",
        df.to_csv(index=False).encode(),
        "housing.csv",
        "text/csv"
    )

elif page == "ℹ️ About":
    st.subheader("ℹ️ About Project")
    st.markdown("""
### Objective
Predict residential house prices using supervised machine learning.

### Workflow
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Preprocessing
6. Train/Test Split
7. Regression Modeling
8. Model Evaluation
9. Interactive Prediction

### Models
- Linear Regression
- Gradient Boosting Regressor
- Random Forest Regressor

### Evaluation
- MAE
- RMSE
- R² Score

### Technology
Python • Pandas • NumPy • Scikit-learn • Plotly • Streamlit
""")