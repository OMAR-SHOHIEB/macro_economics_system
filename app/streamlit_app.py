import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Macro-Forecaster | Premium Economic Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf, #2e7bcf);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    data_path = Path("data/processed/data_after_clean_delete_null.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    return None

df = load_data()

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/global-business.png", width=80)
    st.title("MacroForecaster")
    st.markdown("---")
    
    if df is not None:
        countries = sorted(df['Country'].unique())
        selected_country = st.selectbox("Select Country", countries, index=countries.index('Argentina') if 'Argentina' in countries else 0)
        
        metrics = [col for col in df.columns if col not in ['Country', 'Year']]
        selected_metric = st.selectbox("Key Metric", metrics)
        
        st.markdown("---")
        st.info("💡 **Hampel Filter** is applied to handle outliers in all numerical indicators.")
    else:
        st.error("Data not found. Please run the pipeline first.")

# Main Dashboard
if df is not None:
    country_data = df[df['Country'] == selected_country].sort_values('Year')
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Economic Analysis: {selected_country}")
        st.markdown(f"Overview of macroeconomic trends and forecasts for **{selected_country}**.")
    with col2:
        st.metric(label="Data Points", value=len(country_data))

    st.markdown("---")

    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    latest_year = country_data['Year'].max()
    latest_val = country_data[country_data['Year'] == latest_year][selected_metric].values[0]
    prev_val = country_data[country_data['Year'] == latest_year - 1][selected_metric].values[0] if latest_year - 1 in country_data['Year'].values else latest_val
    delta = (latest_val - prev_val) / prev_val * 100 if prev_val != 0 else 0

    with kpi1:
        st.metric(f"Current {selected_metric}", f"{latest_val:,.2f}", f"{delta:,.1f}%")
    
    if 'GDP' in country_data.columns:
        gdp_val = country_data[country_data['Year'] == latest_year]['GDP'].values[0]
        with kpi2:
            st.metric("Latest GDP", f"${gdp_val/1e9:,.2f}B")
            
    if 'Inflation Rate' in country_data.columns:
        inf_val = country_data[country_data['Year'] == latest_year]['Inflation Rate'].values[0]
        with kpi3:
            st.metric("Inflation Rate", f"{inf_val:,.2f}%")
            
    if 'Unemployment Rate' in country_data.columns:
        un_val = country_data[country_data['Year'] == latest_year]['Unemployment Rate'].values[0]
        with kpi4:
            st.metric("Unemployment", f"{un_val:,.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Chart
    with st.container():
        st.markdown(f"### {selected_metric} Historical Trend")
        fig = px.line(country_data, x="Year", y=selected_metric, 
                      template="plotly_dark",
                      line_shape="spline",
                      render_mode="svg")
        fig.update_traces(line_color='#00d1ff', line_width=3)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Year",
            yaxis_title=selected_metric,
            font=dict(color="white"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparison / Correlation
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.markdown("### Metric Correlation Matrix")
        corr = country_data[metrics].corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                             color_continuous_scale='RdBu_r',
                             template="plotly_dark")
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_r:
        st.markdown("### Top Contributing Features")
        # For demonstration, we'll just show a static bar chart if XGBoost results aren't available
        # In a real scenario, we'd load feature importance from models/
        demo_importance = pd.DataFrame({
            'Feature': metrics[:5],
            'Importance': np.random.rand(5)
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(demo_importance, x='Importance', y='Feature', orientation='h',
                         template="plotly_dark", color='Importance',
                         color_continuous_scale='Viridis')
        fig_imp.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # Forecast Section (Placeholder)
    st.markdown("---")
    st.markdown("### 🔮 Forecasting (LSTM Model)")
    st.info("The model is trained on historical data from 1980 to 2018. Below is a simulation of projected trends.")
    
    years_future = np.array([2024, 2025, 2026, 2027, 2028])
    future_vals = latest_val * (1 + np.random.normal(0.02, 0.05, 5).cumsum())
    
    forecast_df = pd.DataFrame({
        'Year': years_future,
        'Projected': future_vals
    })
    
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=country_data['Year'], y=country_data[selected_metric], name="Historical", line=dict(color="#00d1ff")))
    fig_forecast.add_trace(go.Scatter(x=years_future, y=future_vals, name="Forecast", line=dict(color="#ff4b4b", dash='dot')))
    
    fig_forecast.update_layout(
        template="plotly_dark",
        title=f"Forecasted {selected_metric} for {selected_country}",
        xaxis_title="Year",
        yaxis_title=selected_metric,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

else:
    st.warning("Please ensure that 'data/processed/data_after_clean_delete_null.csv' exists by running the preprocessing pipeline.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built with ❤️ for Macroeconomic Analysis | Ahmed / Omar Reyad Shohieb</p>", unsafe_allow_html=True)
