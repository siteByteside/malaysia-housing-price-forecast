# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import datetime
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

st.set_page_config(page_title="Urban Housing — Historical & Forecast", 
                   page_icon="favicon.png",
                   layout="wide"
                   )

# -------------------------
# Config / Theme variables
# -------------------------
PLOTLY_TEMPLATE = "plotly_dark"  # dark theme
INDIGO = "#3f51b5"
INDIGO_LIGHT = "#6573c3"
ACCENT = "#00bfa5"

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed reading {path}: {e}")
        return None

def standardize_cols(df):
    """Lowercase and strip columns to reduce mismatch issues."""
    if df is None: 
        return None
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_year_int(df, col='Year'):
    if df is None or col not in df.columns:
        return df
    try:
        df[col] = df[col].astype(int)
    except:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        except:
            pass
    return df

# ---- INSIGHTS UI COMPONENT ----
def insights_box(title, items):
    st.markdown(f"### 🔍 {title}")

    html = """
    <div style='
        padding: 18px;
        border: 1px solid #555;
        border-radius: 12px;
        margin-bottom: 20px;
        background-color: rgba(255, 255, 255, 0.08);
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
    '>
        <ul style='margin-left: 20px;'>
    """

    for item in items:
        html += f"<li style='margin-bottom: 6px;'>{item}</li>"

    html += "</ul></div>"

    st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Load data (adjust paths if needed)
# -------------------------
DATA_DIR = Path(".")  # current folder
cleaned_path = DATA_DIR / "historical_housing_2015_2024.csv"
avg_price_path = DATA_DIR / "average_house_price(2015-2029).csv"
avg_types_path = DATA_DIR / "average_house_types(2015-2029).csv"
forecast_path = DATA_DIR / "forecasted_housing_2025_2029.csv"

def file_last_updated(path):
    if Path(path).exists():
        ts = os.path.getmtime(path)
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    return "N/A"

updated_cleaned = file_last_updated(cleaned_path)
updated_price = file_last_updated(avg_price_path)
updated_types = file_last_updated(avg_types_path)
updated_forecast = file_last_updated(forecast_path)

cleaned_df = standardize_cols(load_csv(cleaned_path))
avg_price_df = standardize_cols(load_csv(avg_price_path))
avg_types_df = standardize_cols(load_csv(avg_types_path))
forecast_df = standardize_cols(load_csv(forecast_path))

# quick column normalizing
if cleaned_df is not None:
    # common expected column names: Year, State, District, House_Type, Average_Price, Demand, Price
    cleaned_df.columns = [c.replace(" ", "_") for c in cleaned_df.columns]
    cleaned_df = ensure_year_int(cleaned_df, 'Year')

if avg_price_df is not None:
    avg_price_df = ensure_year_int(avg_price_df, 'Year')

if avg_types_df is not None:
    avg_types_df = ensure_year_int(avg_types_df, 'Year')

if forecast_df is not None:
    forecast_df = ensure_year_int(forecast_df, 'Year')

    # --- FIX STATE NAME VARIATIONS HERE ---
    forecast_df["State"] = forecast_df["State"].replace({
        "Pulau Pinang": "Penang"
    })
    
# -------------------------
# Sidebar - downloads & navigation
# -------------------------
st.sidebar.title("Final Year Project — Housing Dashboard")
st.sidebar.markdown("Files available:")
st.sidebar.markdown("**📅 Data Last Updated:**")
st.sidebar.write(f"- Historical data: {updated_cleaned}")
st.sidebar.write(f"- Average price: {updated_price}")
st.sidebar.write(f"- House types: {updated_types}")
st.sidebar.write(f"- Forecasted data: {updated_forecast}")
st.sidebar.markdown("---")
for p in [cleaned_path, avg_price_path, avg_types_path, forecast_path]:
    if Path(p).exists():
        st.sidebar.download_button(label=f"Download {Path(p).name}", data=Path(p).read_bytes(), file_name=Path(p).name)

page = st.sidebar.radio("Navigate", ["Overview — Historical", "Forecast (2025-2029)", "About / Notes"])

# New Function: Generate PDF report for forecast
def generate_forecast_pdf(selected_states, year_range, summary_df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Malaysia Housing Price Forecast Report")

    # Subtitle
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"States Selected: {', '.join(selected_states)}")
    c.drawString(50, height - 100, f"Years: {year_range[0]} - {year_range[1]}")

    # Summary section
    c.drawString(50, height - 140, "Forecast Summary (Average Price by State):")

    y = height - 160
    for _, row in summary_df.iterrows():
        text = f"{row['State']}: RM {row['AveragePrice']:.2f}"
        c.drawString(60, y, text)
        y -= 20

    # Footer
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 40, "Generated via Malaysia Housing Forecast Dashboard — © 2025 Denise Choo")

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------
# Page: Overview — Historical
# -------------------------
if page == "Overview — Historical":
    st.title("Historical Urban Housing Info (2015-2024)")
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("**Filters**")
        # default state list from cleaned_df if present else top 4 states
        if cleaned_df is not None and "State" in cleaned_df.columns:
            states = sorted(cleaned_df["State"].dropna().unique().tolist())
        else:
            states = ["Kuala Lumpur", "Johor", "Selangor", "Penang"]
        selected_states = st.multiselect("Select states", options=states, default=states)

        # year slider
        if cleaned_df is not None and "Year" in cleaned_df.columns:
            min_y = int(cleaned_df["Year"].min())
            max_y = int(cleaned_df["Year"].max())
            year_range = st.slider("Year range", min_value=min_y, max_value=max_y, value=(min_y, max_y))
        else:
            year_range = (2015, 2024)

    with col2:
        st.markdown("### Historical Average House Price by State")
        # If avg_price_df is only overall average by year, compute state average from cleaned_df
        if avg_price_df is not None and set(["Year","Average_Price","State"]).issubset(avg_price_df.columns):
            avg_state_df = avg_price_df.rename(columns={"Average_Price":"AveragePrice"})
        else:
            if cleaned_df is not None and set(["Year","State"]).issubset(cleaned_df.columns):
                # choose a column that represents price
                price_cols = [c for c in cleaned_df.columns if "price" in c.lower()]
                if price_cols:
                    price_col = price_cols[0]
                    avg_state_df = (cleaned_df.loc[cleaned_df["State"].isin(selected_states)]
                                    .query("Year >= @year_range[0] and Year <= @year_range[1]")
                                    .groupby(["Year","State"], as_index=False)[price_col].mean()
                                    .rename(columns={price_col:"AveragePrice"}))
                else:
                    st.warning("No price column found in cleaned_housing_data.csv to compute average by state.")
                    avg_state_df = pd.DataFrame(columns=["Year","State","AveragePrice"])
            else:
                st.warning("No data available to produce state-level average price plot.")
                avg_state_df = pd.DataFrame(columns=["Year","State","AveragePrice"])

        if not avg_state_df.empty:
            fig_state = px.line(avg_state_df[avg_state_df["State"].isin(selected_states)],
                                x="Year", y="AveragePrice", color="State", markers=True,
                                title="Average Price by State",
                                template=PLOTLY_TEMPLATE)
            fig_state.update_traces(line=dict(width=3))
            st.plotly_chart(fig_state, use_container_width=True)
            # ---- KEY INSIGHTS: HISTORICAL ----
            hist_df = avg_state_df[avg_state_df["State"].isin(selected_states)]

            insights = []

            # CAGR helper
            def cagr(start, end, years):
                try:
                    return ((end / start) ** (1 / years) - 1) * 100
                except:
                    return None
            # 1. Growth insights (CAGR)
            for state in selected_states:
                df_s = hist_df[hist_df["State"] == state].sort_values("Year")
                if len(df_s) > 1:
                    start = df_s["AveragePrice"].iloc[0]
                    end = df_s["AveragePrice"].iloc[-1]
                    years = df_s["Year"].iloc[-1] - df_s["Year"].iloc[0]
                    g = cagr(start, end, years)
                    if g is not None:
                        insights.append(f"{state}: {g:.2f}% annual growth (CAGR)")
            # 2. Highest price in latest year
            latest_year = hist_df["Year"].max()
            df_latest = hist_df[hist_df["Year"] == latest_year]

            # Only show "highest state" if multiple states are selected
            if len(selected_states) > 1 and not df_latest.empty:
                top_state = df_latest.loc[df_latest["AveragePrice"].idxmax()]["State"]
                insights.append(f"{top_state} has the highest average price in {latest_year}")

            # Render insights box
            insights_box("Key Historical Insights", insights)
        else:
            st.info("State average price chart unavailable due to missing data.")

        st.markdown("### Historical Average House Price by House Type")
        # house-type bar chart for selected year
        sel_year = st.selectbox("Select year to view house type average", options=list(range(year_range[0], year_range[1]+1)))
        if avg_types_df is not None and set(["Year","House_Type","Average_Price"]).issubset(avg_types_df.columns):
            df_ht = avg_types_df[avg_types_df["Year"] == sel_year]
            fig_ht = px.bar(df_ht, x="House_Type", y="Average_Price", title=f"Average Price by House Type — {sel_year}",
                            template=PLOTLY_TEMPLATE)
            st.plotly_chart(fig_ht, use_container_width=True)
        else:
            # fallback compute from cleaned_df
            if cleaned_df is not None and set(["Year","House_Type"]).issubset(cleaned_df.columns):
                price_cols = [c for c in cleaned_df.columns if "price" in c.lower()]
                if price_cols:
                    price_col = price_cols[0]
                    df_ht = (cleaned_df[cleaned_df["Year"]==sel_year]
                             .groupby("House_Type", as_index=False)[price_col].mean()
                             .rename(columns={price_col:"Average_Price"}))
                    fig_ht = px.bar(df_ht, x="House_Type", y="Average_Price", title=f"Average Price by House Type — {sel_year}",
                                    template=PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_ht, use_container_width=True)
                else:
                    st.info("No house-type price data available.")
            else:
                st.info("House-type average price data unavailable.")

    st.markdown("---")
    st.markdown("### Historical Average House Price by District")
    if cleaned_df is not None and "District" in cleaned_df.columns:
        # allow year selection for treemap
        treemap_year = st.selectbox("Select year for district treemap", options=list(range(year_range[0], year_range[1]+1)), key="treemap_year")
        price_cols = [c for c in cleaned_df.columns if "price" in c.lower()]
        if price_cols:
            price_col = price_cols[0]
            treemap_df = (cleaned_df[cleaned_df["Year"]==treemap_year]
                          .groupby(["State","District"], as_index=False)[price_col].mean()
                          .rename(columns={price_col:"AveragePrice"}))
            # restrict to selected states
            treemap_df = treemap_df[treemap_df["State"].isin(selected_states)]
            if treemap_df.empty:
                st.info("No district data for selected year / states.")
            else:
                fig_tree = px.treemap(treemap_df, path=["State","District"], values="AveragePrice",
                                     title=f"District Average Price — {treemap_year}", template=PLOTLY_TEMPLATE)
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.info("No price field found in the cleaned dataset to build district treemap.")
    else:
        st.info("No district column found in cleaned dataset.")

# -------------------------
# Page: Forecast
# -------------------------
elif page == "Forecast (2025-2029)":
    st.title("Forecasted Urban Housing Info (2025-2029)")

    if forecast_df is None:
        st.error("Forecast file (forecasted_prices_2025_2029.csv) not found or unreadable.")
    else:
        # Ensure correct price column name
        if "Predicted_House_Price" not in forecast_df.columns:
            cand = [c for c in forecast_df.columns if "pred" in c.lower() and "price" in c.lower()]
            if cand:
                forecast_df = forecast_df.rename(columns={cand[0]: "Predicted_House_Price"})

        # Keep only the 4 urban states
        kept_states = ["Kuala Lumpur", "Johor", "Selangor", "Penang"]
        if "State" in forecast_df.columns:
            forecast_df = forecast_df[forecast_df["State"].isin(kept_states)]

        forecast_df = ensure_year_int(forecast_df, "Year")

        required_cols = ["Year", "State", "Predicted_House_Price"]
        if set(required_cols).issubset(forecast_df.columns):

            # Prepare grouped forecast data
            forecast_group = (
                forecast_df
                .groupby(["Year", "State"], as_index=False)["Predicted_House_Price"]
                .mean()
                .rename(columns={"Predicted_House_Price": "AveragePrice"})
            )

            years = sorted(forecast_group["Year"].unique())
            min_y, max_y = min(years), max(years)

            # -----------------------------
            # TWO-COLUMN LAYOUT (MATCHES HISTORICAL)
            # -----------------------------
            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown("**Filters**")

                # State multi-select
                sel_states = st.multiselect(
                    "Select States",
                    options=kept_states,
                    default=kept_states
                )

                # Year slider (forecast)
                year_range = st.slider(
                    "Year range",
                    min_value=min_y,
                    max_value=max_y,
                    value=(min_y, max_y)
                )

            with col2:
                st.markdown("### Forecasted Average House Price by State")
                # Apply filters
                plot_df = forecast_group[
                    (forecast_group["Year"] >= year_range[0]) &
                    (forecast_group["Year"] <= year_range[1]) &
                    (forecast_group["State"].isin(sel_states))
                ]

                if plot_df.empty:
                    st.info("No forecast data for selected years/states.")
                else:
                    figf = px.line(
                        plot_df, x="Year", y="AveragePrice",
                        color="State", markers=True,
                        title="Average Price by State",
                        template=PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(figf, use_container_width=True)
                    # ---- KEY INSIGHTS: FORECAST ----
                    f_df = plot_df  # already filtered by state + year

                    insights_f = []

                    # 1. Growth from first → last year
                    for state in sel_states:
                        df_s = f_df[f_df["State"] == state].sort_values("Year")
                        if len(df_s) > 1:
                            start = df_s["AveragePrice"].iloc[0]
                            end = df_s["AveragePrice"].iloc[-1]
                            growth = ((end - start) / start) * 100
                            insights_f.append(f"{state}: projected {growth:.1f}% increase across selected years")

                    # 2. Highest + lowest in end year
                    end_year = f_df["Year"].max()
                    df_end = f_df[f_df["Year"] == end_year]

                    if len(sel_states) > 1 and not df_end.empty:
                        highest = df_end.loc[df_end["AveragePrice"].idxmax()]["State"]
                        lowest = df_end.loc[df_end["AveragePrice"].idxmin()]["State"]

                        insights_f.append(f"Highest predicted average price in {end_year}: {highest}")
                        insights_f.append(f"Lowest predicted average price in {end_year}: {lowest}")

                    # Render insights panel
                    insights_box("Key Forecast Insights", insights_f)

                    # -----------------------------
                    # HOUSE-TYPE FORECAST SECTION
                    # -----------------------------
                    st.markdown("### Forecasted Average House Price by House Type")

                    sel_year = st.selectbox(
                        "Select year to inspect house-type forecasts",
                        options=years,
                        index=0
                    )

                    if set(["Year", "House_Type", "Predicted_House_Price"]).issubset(forecast_df.columns):
                        df_ht = (
                            forecast_df[forecast_df["Year"] == sel_year]
                            .groupby("House_Type", as_index=False)["Predicted_House_Price"]
                            .mean()
                            .rename(columns={"Predicted_House_Price": "AveragePrice"})
                        )

                        fig_htf = px.bar(
                            df_ht,
                            x="House_Type",
                            y="AveragePrice",
                            title=f"Average Price by House Type — {sel_year}",
                            template=PLOTLY_TEMPLATE
                        )
                        st.plotly_chart(fig_htf, use_container_width=True)
                        # ---- PDF REPORT GENERATION ----
                        st.markdown("### Download Forecast Report (PDF)")
                        # Build summary table
                        summary_df = (
                            plot_df.groupby("State", as_index=False)["AveragePrice"]
                                .mean()
                        )
                        pdf_buffer = generate_forecast_pdf(sel_states, year_range, summary_df)
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name="housing_forecast_report.pdf",
                            mime="application/pdf"
                        )

                    else:
                        st.info("House type forecasts not available in forecast file.")
        else:
            st.error("Forecast file lacks required columns ('Year','State','Predicted_House_Price').")



# -------------------------
# Page: About / Notes
# -------------------------
else:
    st.title("About this project")
    st.markdown("""
### 🧠 **Model Overview**
This dashboard is powered by a predictive machine learning model (`rf_tuned_22`) built using a **Random Forest Regressor**.
The model was trained on **Malaysia urban housing data from 2015–2024**, focusing on Kuala Lumpur, Selangor, Johor, and Penang.
It forecasts average house prices for **2025–2029**.

---
### 🔑 **Key Features the Model Relies On**
The model uses a range of property, location, and demographic features.
Among all inputs, the **top 3 most influential features** (based on feature importance) are:


1. **District** *(location signal — strongest predictor)*
2. **House Type** *(terrace, condo, landed, etc.)*
3. **Population** *(population scale & urban density patterns)*

These three features together drive a significant portion of the model’s predictive power.


---


### ⚙️ **Model Training & Tuning**
- **Training approach:** time-series based
- Model trained on **≤ 2021 data**
- Tested on **> 2021 data**
- Ensures evaluation simulates real future forecasting, not random splits
- **Tuning methods used:**
- **Manual tuning**
- **RandomizedSearchCV** for systematic hyperparameter exploration


---
                
### 📏 **Model Performance**
To evaluate accuracy, we measured prediction error on the test period (post-2021):


- **MAE (Mean Absolute Error): ≈ 20%**
- Predictions are on average within **±20%** of actual historical prices.

---


### ⚠️ **Limitations to be Aware Of**
- Forecasts assume historical patterns continue without major shocks.
- The model does not include policy changes, macroeconomic shifts, or major development projects.
- Predictions are **state-level & house-type-level**, not district-level.
- Uncertainty intervals not yet included (planned improvement).


---
                
### 📱 **Contact & Links**
- Email: denisechoo8236@gmail.com
- GitHub: https://github.com/siteByteside
""")


st.markdown("---")

# ---- FOOTER ----
footer_html = """
    <div style="text-align: center; padding-top: 10px;">
        <span style="font-size: 14px; color: gray;">
            Developed by Denise Choo © 2025<br>
            Malaysia Housing Forecast Dashboard
        </span>
    </div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
