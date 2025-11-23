# Malaysia Housing Price Forecast (2015–2029)

A data-driven, interactive Streamlit dashboard analyzing and forecasting Malaysia’s urban housing market.  
This project focuses on four key urban states — **Kuala Lumpur, Selangor, Penang, and Johor** — using:

- **Historical housing data (2015–2024)** from NAPIC, DOSM, and BNM  
- **Machine-learning predictions (2025–2029)** generated using a tuned Random Forest model (`rf_tuned_22`)

---

## 🚀 Features

### 🔹 Historical Insights (2015–2024)
- Average house price trends by **state**, **district**, and **house type**
- Treemap visualizations for district-level price distribution  
- Multi-year sliders and interactive filtering  
- Clean, modern Plotly visualizations

### 🔹 Forecasting (2025–2029)
- Predicted average house prices by **state**  
- Forecasted trends for **multiple years**  
- House-type level forecast breakdown  
- Based on ML model trained on 10 years of real housing market data

### 🔹 Downloadable Data
- Historical dataset (2015–2024)
- Forecasted dataset (2025–2029)

---

## 🧠 Methodology
- **Model:** Random Forest Regressor (tuned version `rf_tuned_22`)  
- **Target:** Average housing price  
- **Features include:** state, district, house type, socioeconomic and property attributes  
- **Training data:** 2015–2024  
- **Forecast horizon:** 2025–2029

---

## 📁 Project Structure
malaysia-housing-price-forecast/
│
├── app.py
├── requirements.txt
│
├── historical_housing_2015_2024.csv
├── average_house_price(2015-2029).csv
├── average_house_types(2015-2029).csv
├── forecasted_housing_2025_2029.csv

## ▶️ Running the App Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
streamlit run app.py
```
