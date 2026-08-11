# 🏠 House Price Prediction System

Complete Streamlit ML project using the uploaded Kaggle-style housing dataset.

## Dataset
The project uses the uploaded dataset saved as `data/housing.csv`.

- Rows: 4600
- Columns: 18
- Target: `price`

### Features
date, price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, street, city, statezip, country

## Criteria Covered
- Data Cleaning & Preprocessing
- Exploratory Data Analysis
- Feature Engineering / preprocessing
- Linear Regression
- Gradient Boosting
- Random Forest
- MAE, RMSE, R²
- Actual vs Predicted visualization
- Residual analysis
- Interactive Streamlit prediction
- Dataset preview
- Jupyter-ready project structure

## Run

```bash
pip install -r requirements.txt
python -m src.train
streamlit run app/app.py
```

## Project Structure

```text
house_price_prediction/
├── app.py
│  
├── data/
│   └── housing.csv
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   └── train.py
├── requirements.txt
└── README.md
```

> For submission, mention the exact Kaggle dataset/source used by you in your report and repository.
