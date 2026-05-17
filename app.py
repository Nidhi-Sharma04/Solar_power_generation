import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #334155);
        border: 1px solid #f59e0b44;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f59e0b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    .stSlider label { color: #cbd5e1 !important; font-size: 0.85rem; }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .sidebar-header {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #0f172a;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("solar_model.sav", "rb") as f:
        return pickle.load(f)

model = load_model()

FEATURE_NAMES = [
    'temperature_2_m_above_gnd',
    'relative_humidity_2_m_above_gnd',
    'mean_sea_level_pressure_MSL',
    'total_precipitation_sfc',
    'snowfall_amount_sfc',
    'total_cloud_cover_sfc',
    'high_cloud_cover_high_cld_lay',
    'medium_cloud_cover_mid_cld_lay',
    'low_cloud_cover_low_cld_lay',
    'shortwave_radiation_backwards_sfc',
    'wind_speed_10_m_above_gnd',
    'wind_direction_10_m_above_gnd',
    'wind_speed_80_m_above_gnd',
    'wind_direction_80_m_above_gnd',
    'wind_speed_900_mb',
    'wind_direction_900_mb',
    'wind_gust_10_m_above_gnd',
    'angle_of_incidence',
    'zenith',
    'azimuth',
]

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<h1 style='font-size:3.5rem;margin:0'>☀️</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("## Solar Power Generation Predictor")
    st.markdown("<p style='color:#94a3b8;margin-top:-8px'>Random Forest · 20 Meteorological Features · Predict kW output</p>",
                unsafe_allow_html=True)

st.divider()

# ── Sidebar — Input Features ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-header'>⚙️ Input Parameters</div>", unsafe_allow_html=True)

    st.markdown("**🌡️ Atmospheric**")
    temperature        = st.slider("Temperature 2m (°C)",         -10.0, 50.0,  18.0, 0.1)
    relative_humidity  = st.slider("Relative Humidity 2m (%)",      0,   100,    65,   1)
    mean_sea_pressure  = st.slider("Mean Sea Level Pressure (hPa)", 950.0, 1060.0, 1013.0, 0.1)
    total_precip       = st.slider("Total Precipitation (mm)",       0.0,  50.0,   0.0,  0.1)
    snowfall           = st.slider("Snowfall Amount (mm)",           0.0,  50.0,   0.0,  0.1)

    st.markdown("**☁️ Cloud Cover**")
    total_cloud        = st.slider("Total Cloud Cover (%)",          0.0, 100.0,  25.0, 1.0)
    high_cloud         = st.slider("High Cloud Cover (%)",           0,   100,    10,   1)
    medium_cloud       = st.slider("Medium Cloud Cover (%)",         0,   100,    10,   1)
    low_cloud          = st.slider("Low Cloud Cover (%)",            0,   100,     5,   1)

    st.markdown("**🌞 Solar Angles & Radiation**")
    shortwave_rad      = st.slider("Shortwave Radiation (W/m²)",     0.0, 1200.0, 600.0, 1.0)
    angle_incidence    = st.slider("Angle of Incidence (°)",         0.0,  90.0,  30.0,  0.1)
    zenith             = st.slider("Zenith Angle (°)",               0.0,  90.0,  45.0,  0.1)
    azimuth            = st.slider("Azimuth Angle (°)",              0.0, 360.0, 180.0,  0.5)

    st.markdown("**💨 Wind (10m)**")
    wind_speed_10      = st.slider("Wind Speed 10m (m/s)",           0.0,  30.0,   5.0,  0.1)
    wind_dir_10        = st.slider("Wind Direction 10m (°)",         0.0, 360.0, 180.0,  1.0)

    st.markdown("**💨 Wind (80m)**")
    wind_speed_80      = st.slider("Wind Speed 80m (m/s)",           0.0,  40.0,   8.0,  0.1)
    wind_dir_80        = st.slider("Wind Direction 80m (°)",         0.0, 360.0, 180.0,  1.0)

    st.markdown("**💨 Wind (900 mb)**")
    wind_speed_900     = st.slider("Wind Speed 900mb (m/s)",         0.0,  50.0,  10.0,  0.1)
    wind_dir_900       = st.slider("Wind Direction 900mb (°)",       0.0, 360.0, 180.0,  1.0)
    wind_gust          = st.slider("Wind Gust 10m (m/s)",            0.0,  50.0,   8.0,  0.1)

    predict_btn = st.button("⚡ Predict Power Output", use_container_width=True, type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
input_values = [
    temperature, relative_humidity, mean_sea_pressure, total_precip,
    snowfall, total_cloud, high_cloud, medium_cloud, low_cloud,
    shortwave_rad, wind_speed_10, wind_dir_10, wind_speed_80, wind_dir_80,
    wind_speed_900, wind_dir_900, wind_gust, angle_incidence, zenith, azimuth
]

input_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)

# Always run prediction (live update on slider change)
predicted_kw = model.predict(input_df)[0]
predicted_kw = max(0.0, predicted_kw)   # clamp negative edge cases

# ── Result Cards ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{predicted_kw:,.1f}</div>
        <div class='metric-label'>Predicted Power (kW)</div>
    </div>""", unsafe_allow_html=True)

with c2:
    mwh = predicted_kw / 1000
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{mwh:.3f}</div>
        <div class='metric-label'>Power (MW)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    # Assume 3000 kW max capacity for % calculation
    MAX_CAPACITY = 3056.8
    efficiency = min((predicted_kw / MAX_CAPACITY) * 100, 100)
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{efficiency:.1f}%</div>
        <div class='metric-label'>Capacity Utilisation</div>
    </div>""", unsafe_allow_html=True)

with c4:
    # Rough CO₂ offset: 0.82 kg per kWh avoided (coal baseline)
    co2_offset = predicted_kw * 0.82
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{co2_offset:,.0f}</div>
        <div class='metric-label'>CO₂ Offset (g/hr)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Power Gauge Bar ───────────────────────────────────────────────────────────
st.markdown("#### ⚡ Power Output Level")
gauge_pct = min(predicted_kw / MAX_CAPACITY, 1.0)
bar_color = "#22c55e" if gauge_pct > 0.6 else ("#f59e0b" if gauge_pct > 0.3 else "#ef4444")
st.markdown(f"""
<div style="background:#1e293b;border-radius:10px;padding:4px;margin-bottom:1rem">
  <div style="width:{gauge_pct*100:.1f}%;background:{bar_color};
              border-radius:8px;height:26px;
              display:flex;align-items:center;justify-content:center;
              color:#0f172a;font-weight:700;font-size:0.85rem;
              transition:width 0.4s ease;">
    {predicted_kw:,.1f} kW
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input Summary Table ───────────────────────────────────────────────────────
st.markdown("#### 📋 Input Feature Summary")

col_a, col_b = st.columns(2)
labels = [
    "Temperature (°C)", "Relative Humidity (%)", "Sea Level Pressure (hPa)",
    "Total Precipitation (mm)", "Snowfall (mm)", "Total Cloud Cover (%)",
    "High Cloud Cover (%)", "Medium Cloud Cover (%)", "Low Cloud Cover (%)",
    "Shortwave Radiation (W/m²)"
]
values_a = input_values[:10]

labels_b = [
    "Wind Speed 10m (m/s)", "Wind Dir 10m (°)", "Wind Speed 80m (m/s)",
    "Wind Dir 80m (°)", "Wind Speed 900mb (m/s)", "Wind Dir 900mb (°)",
    "Wind Gust 10m (m/s)", "Angle of Incidence (°)", "Zenith (°)", "Azimuth (°)"
]
values_b = input_values[10:]

with col_a:
    df_a = pd.DataFrame({"Feature": labels, "Value": values_a})
    st.dataframe(df_a, use_container_width=True, hide_index=True)

with col_b:
    df_b = pd.DataFrame({"Feature": labels_b, "Value": values_b})
    st.dataframe(df_b, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem'>"
    "Model: Random Forest Regressor · R² = 0.8216 · Trained on 4,213 meteorological samples"
    "</p>",
    unsafe_allow_html=True
)
