# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Urban Housing — Historical & Forecast", layout="wide")

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

# -------------------------
# Load data (adjust paths if needed)
# -------------------------
DATA_DIR = Path(".")  # current folder
cleaned_path = DATA_DIR / "historical_housing_2015_2024.csv"
avg_price_path = DATA_DIR / "average_house_price(2015-2029).csv"
avg_types_path = DATA_DIR / "average_house_types(2015-2029).csv"
forecast_path = DATA_DIR / "forecasted_housing_2025_2029.csv"

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

# -------------------------
# Sidebar - downloads & navigation
# -------------------------
st.sidebar.title("Housing FYP — Dashboard")
st.sidebar.markdown("Files available:")
for p in [cleaned_path, avg_price_path, avg_types_path, forecast_path]:
    if Path(p).exists():
        st.sidebar.download_button(label=f"Download {Path(p).name}", data=Path(p).read_bytes(), file_name=Path(p).name)

page = st.sidebar.radio("Navigate", ["Overview — Historical", "Forecast (2025-2029)", "About / Notes"])

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
**Project**: Predictive ML model (`rf_tuned_22`) for Malaysia urban housing prices — forecasting 2025–2029.  
**Historical data**: 2015–2024 (government sources such as NAPIC, DOSM, BNM).  
**Focus**: urban states — Kuala Lumpur, Johor, Selangor, Penang.  
**Audience / stakeholders**: policymakers, urban housing planners, real-estate developers, researchers.

**Important caveats**
- Forecasts are strategic guidance, not precise transaction prices. External shocks (policy, macro, political) can change outcomes.
- Retrain model whenever new official data is published (ideally annually).
- Include uncertainty intervals for stakeholder use (future improvement).
    """)
    st.markdown("---")
    st.markdown("Contact & Links")
    st.write("- Portfolio:", "https://denisechoo80.wixsite.com/portfolio")
    st.write("- GitHub:", "https://github.com/siteByteside")
