import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import json
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Suppress sklearn/xgboost version mismatch warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ANALYTICS_DIR = PROJECT_ROOT / "analytics"

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Index (AQI) Predictor & Safety Guide",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Theme-Adaptive & High-Contrast)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.1rem;
        margin-bottom: 0.4rem;
    }
    .main-header p {
        color: #e2e8f0 !important;
        font-size: 1.05rem;
        margin: 0;
        line-height: 1.5;
    }

    /* St.Button Primary */
    .stButton>button {
        background: linear-gradient(135deg, #0d9488 0%, #1e3a8a 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.4) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.6) !important;
        color: #ffffff !important;
    }

/* Streamlit Markdown typography can override nested HTML card text colors.
   Scope high-contrast rules to our own cards, without changing tables/plots. */
.stMarkdown .aqi-contrast-card,
.stMarkdown .aqi-contrast-card p,
.stMarkdown .aqi-contrast-card h2,
.stMarkdown .aqi-contrast-card h3,
.stMarkdown .aqi-contrast-card div,
.stMarkdown .aqi-contrast-card b,
.stMarkdown .aqi-contrast-card span:not(.aqi-bucket-pill):not(.aqi-accent),
.stMarkdown .aqi-contrast-card strong {
    color: #f8fafc !important;
}
.stMarkdown .aqi-contrast-card .aqi-accent {
    color: #fbbf24 !important;
}
.stMarkdown .aqi-contrast-card .aqi-bucket-pill {
    color: #172033 !important;
}
.stMarkdown .aqi-contrast-card .aqi-muted {
    color: #dbeafe !important;
}
@media (max-width: 720px) {
    .stMarkdown .aqi-contrast-card { padding: 1rem !important; }
    .stMarkdown .aqi-contrast-card h2 { font-size: 1.65rem !important; }
    .stMarkdown .aqi-contrast-card h3 { font-size: 1.05rem !important; }
}

/* Executive KPI cards: accessible, responsive, and no truncated model name. */
.aqi-kpi-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:12px 0 8px;}
.aqi-kpi {position:relative;min-width:0;border:1px solid #dce5ef;border-radius:16px;padding:18px 19px;background:linear-gradient(155deg,#ffffff 0%,#f6f9fe 100%);box-shadow:0 5px 17px rgba(15,23,42,.075);overflow-wrap:anywhere;}
.aqi-kpi.best {background:linear-gradient(135deg,#173b80,#087f81);border-color:#157b89;box-shadow:0 8px 24px rgba(13,98,125,.18);}
.aqi-kpi .kpi-caption {font-size:.83rem;font-weight:650;color:#52627b;line-height:1.35;margin:0 0 9px;}
.aqi-kpi .kpi-value {font-size:clamp(1.45rem,2.05vw,2.1rem);font-weight:800;letter-spacing:-.035em;color:#142a49;line-height:1.18;margin:0;}
.aqi-kpi.best .kpi-caption {color:#dbeafe;}
.aqi-kpi.best .kpi-value {font-size:clamp(1.3rem,1.8vw,1.85rem);color:#ffffff;white-space:normal;overflow-wrap:normal;}
.aqi-kpi .kpi-foot {font-size:.75rem;color:#657991;margin-top:10px;}
.aqi-kpi.best .kpi-foot {color:#c6f3ed;}
@media(max-width:1000px){.aqi-kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr));}}
@media(max-width:550px){.aqi-kpi-grid{grid-template-columns:1fr 1fr;gap:9px}.aqi-kpi{padding:12px}.aqi-kpi .kpi-value,.aqi-kpi.best .kpi-value{font-size:1.25rem}.aqi-kpi .kpi-caption{font-size:.74rem}}
/* Sidebar preset actions use two rows rather than cutting words with ellipses. */
section[data-testid="stSidebar"] .stButton > button {white-space:normal!important;min-height:44px;padding:.55rem .7rem!important;font-size:.91rem!important;line-height:1.3!important;}

@media (max-width: 720px) {
    .main-header {padding: 1.05rem 1.1rem; border-radius: 12px;}
    .main-header h1 {font-size: 1.45rem;}
    .main-header p {font-size: 0.9rem;}
    .stButton>button {font-size: 0.93rem !important; padding: 0.5rem 0.7rem !important;}
    [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
FEATURE_COLS = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene']

DEFAULT_VALUES = {
    'PM2.5': 53.0,
    'PM10': 102.0,
    'NO': 10.0,
    'NO2': 22.5,
    'NOx': 25.0,
    'NH3': 18.0,
    'CO': 0.95,
    'SO2': 9.5,
    'O3': 31.0,
    'Benzene': 1.0,
    'Toluene': 3.0
}

# Pre-configured Presets
PRESETS = {
    "Select a Scenario Preset...": None,
    "🌱 Pristine Himalayan / Forest (Good)": {
        'PM2.5': 15.0, 'PM10': 28.0, 'NO': 4.0, 'NO2': 9.5, 'NOx': 11.0,
        'NH3': 6.0, 'CO': 0.35, 'SO2': 5.0, 'O3': 18.0, 'Benzene': 0.25, 'Toluene': 0.60
    },
    "🌤️ Clean Suburban Morning (Satisfactory)": {
        'PM2.5': 42.0, 'PM10': 75.0, 'NO': 8.5, 'NO2': 21.0, 'NOx': 24.0,
        'NH3': 15.0, 'CO': 0.70, 'SO2': 8.5, 'O3': 28.0, 'Benzene': 0.90, 'Toluene': 2.50
    },
    "🏙️ Moderate Traffic City (Moderate)": {
        'PM2.5': 85.0, 'PM10': 145.0, 'NO': 18.0, 'NO2': 38.0, 'NOx': 42.0,
        'NH3': 26.0, 'CO': 1.35, 'SO2': 15.0, 'O3': 42.0, 'Benzene': 2.80, 'Toluene': 7.20
    },
    "🏭 Industrial & Heavy Congestion (Poor)": {
        'PM2.5': 155.0, 'PM10': 235.0, 'NO': 38.0, 'NO2': 62.0, 'NOx': 72.0,
        'NH3': 40.0, 'CO': 2.60, 'SO2': 26.0, 'O3': 56.0, 'Benzene': 6.50, 'Toluene': 16.0
    },
    "🌫️ Winter Smog & Thermal Inversion (Very Poor)": {
        'PM2.5': 240.0, 'PM10': 360.0, 'NO': 65.0, 'NO2': 88.0, 'NOx': 105.0,
        'NH3': 58.0, 'CO': 4.20, 'SO2': 42.0, 'O3': 74.0, 'Benzene': 11.0, 'Toluene': 28.0
    },
    "🚨 Hazardous Air Quality Emergency (Severe)": {
        'PM2.5': 450.0, 'PM10': 620.0, 'NO': 120.0, 'NO2': 140.0, 'NOx': 175.0,
        'NH3': 90.0, 'CO': 8.00, 'SO2': 65.0, 'O3': 105.0, 'Benzene': 20.0, 'Toluene': 52.0
    }
}

# LabelEncoder alphabetical ordering from training notebook
BUCKET_MAPPING = {
    0: 'Good',
    1: 'Moderate',
    2: 'Poor',
    3: 'Satisfactory',
    4: 'Severe',
    5: 'Very Poor'
}

# CPCB Standard Thresholds, Colors & Medically-backed Precautions
BUCKET_CONFIG = {
    'Good': {
        'color': '#10B981',
        'text_color': '#ffffff',
        'range': '0 - 50',
        'severity': 'Minimal Health Impact',
        'icon': '🌱',
        'description': 'Air quality is considered satisfactory, and air pollution poses little or no risk to public health.',
        'dos': [
            "Outdoor sports, cycling, and vigorous cardio are strongly encouraged for all age groups.",
            "Open all doors and windows to ensure generous natural fresh-air home and office ventilation.",
            "Take advantage of clear skies for morning walks, children's park activities, and outdoor gardening.",
            "Maintain indoor and balcony plants to sustain clean ambient micro-environments."
        ],
        'donts': [
            "Avoid creating open fires by burning dry leaves, garden waste, garbage, or plastics.",
            "Avoid leaving car engines idling unnecessarily in parking areas or driveways.",
            "Do not miss the opportunity to engage in outdoor recreational and fitness routines!"
        ],
        'vulnerable': "Safe for everyone, including infants, elderly citizens, and people with heart or respiratory conditions.",
        'mask': "No protective mask required under these clean atmospheric conditions.",
        'purifier': "Air purifiers are not needed; natural outdoor air is clean and beneficial."
    },
    'Satisfactory': {
        'color': '#84CC16',
        'text_color': '#1f2937',
        'range': '51 - 100',
        'severity': 'Minor Impact on Sensitive Individuals',
        'icon': '🌤️',
        'description': 'Air quality is acceptable; however, there may be minor breathing discomfort to unusually sensitive individuals.',
        'dos': [
            "Safe for almost all daily outdoor activities, commuting, sports, and school activities.",
            "Unusually sensitive individuals with severe asthma should keep quick-relief rescue inhalers accessible.",
            "Ventilate indoor living spaces during midday when ambient temperature and wind dispersion are favorable.",
            "Stay adequately hydrated throughout the day with clean water and fresh fluids."
        ],
        'donts': [
            "Unusually sensitive people should not perform hours of continuous heavy exertion directly beside heavy-traffic highways.",
            "Do not smoke tobacco or burn biomass fuel indoors in enclosed spaces.",
            "Avoid burning garden waste, crop residues, or open trash."
        ],
        'vulnerable': "Generally safe; hypersensitive individuals with reactive airway disease may notice minor throat dryness.",
        'mask': "Not needed for the general public. Sensitive individuals may opt for a basic mask near dense vehicular congestion.",
        'purifier': "Not strictly necessary. Natural cross-ventilation remains safe and recommended."
    },
    'Moderate': {
        'color': '#F59E0B',
        'text_color': '#1f2937',
        'range': '101 - 200',
        'severity': 'Moderate Breathing Discomfort',
        'icon': '⚠️',
        'description': 'Breathing discomfort likely for children, older adults, and people with lung disease, asthma, and heart conditions.',
        'dos': [
            "Children, elderly citizens, and individuals with respiratory conditions should moderate prolonged outdoor exertion.",
            "Schedule heavy outdoor workouts for late mornings or afternoons when smog dispersion is higher.",
            "Keep indoor air clean: keep windows facing busy highways closed during peak rush hours.",
            "Stay well hydrated; drink warm water, herbal teas, or antioxidant-rich fluids.",
            "Keep prescribed asthma inhalers and cardiac medications ready at hand."
        ],
        'donts': [
            "Do not perform strenuous endurance exercise (e.g., marathon training) directly alongside congested roadways.",
            "Do not smoke indoors, light incense sticks (agarbatti), or burn candles in unventilated rooms.",
            "Avoid dry dusting or sweeping inside homes—use damp wet mops instead to trap fine particles.",
            "Do not ignore symptoms like persistent coughing, throat irritation, or wheezing."
        ],
        'vulnerable': "People with asthma, COPD, bronchitis, hypertension, or heart disease should minimize prolonged outdoor exertion.",
        'mask': "Recommended for sensitive individuals during rush hour commutes (surgical or KN95).",
        'purifier': "Beneficial in bedrooms for sensitive individuals, especially overnight."
    },
    'Poor': {
        'color': '#F97316',
        'text_color': '#ffffff',
        'range': '201 - 300',
        'severity': 'Breathing Discomfort to Most People',
        'icon': '😷',
        'description': 'Breathing discomfort to most people on prolonged exposure; significant respiratory distress for sensitive groups.',
        'dos': [
            "Wear a certified, well-fitted N95 or N99 respirator mask whenever stepping outdoors.",
            "Vulnerable groups (children, seniors, asthma/COPD patients) should remain indoors as much as possible.",
            "Run indoor HEPA air purifiers inside living and sleeping quarters.",
            "Keep doors and windows firmly shut during early morning and late night inversion hours.",
            "Practice steam inhalation with saline or eucalyptus oil after returning from outdoors.",
            "Include vitamin C, vitamin E, and antioxidant-rich foods in your diet to counter oxidative stress."
        ],
        'donts': [
            "Strictly avoid early morning jogs, cycling, or outdoor athletic sports.",
            "Do not drive with car windows rolled down; set car air conditioning to internal recirculate mode.",
            "Do not allow young children to play outdoor sports in dusty playgrounds during peak smog hours.",
            "Do not burn firecrackers, wood, kerosene heaters, or mosquito coils inside closed rooms.",
            "Avoid deep frying or heavy grilling indoors without an active kitchen exhaust hood."
        ],
        'vulnerable': "High risk for children, elderly, asthmatics, and heart patients. May trigger acute asthma attacks and angina.",
        'mask': "N95 / FFP2 mask strongly recommended for anyone spending more than 15 minutes outdoors.",
        'purifier': "HEPA air purifier recommended in active indoor rooms; keep door gaps covered."
    },
    'Very Poor': {
        'color': '#EF4444',
        'text_color': '#ffffff',
        'range': '301 - 400',
        'severity': 'Respiratory Illness on Prolonged Exposure',
        'icon': '🚨',
        'description': 'Triggers respiratory illness on prolonged exposure. Severe effect on people with existing lung or heart diseases.',
        'dos': [
            "Stay strictly indoors with windows and doors completely closed and draft-sealed.",
            "Mandatory certified N95/N99 mask if stepping outside is unavoidable for critical emergencies.",
            "Run HEPA air purifiers continuously on medium/high speed in all occupied rooms.",
            "Use wet mopping exclusively to clean floors and damp microfiber cloths for surface dusting.",
            "Stay well hydrated; consume warm fluids and soothing honey-ginger infusions.",
            "Monitor blood oxygen (SpO2) and peak flow meter readings if you have chronic asthma or COPD.",
            "Contact a healthcare professional promptly if wheezing, chest tightness, or shortness of breath occurs."
        ],
        'donts': [
            "ABSOLUTELY NO outdoor workouts, morning walks, running, or outdoor physical labor.",
            "Do not send children to outdoor playgrounds; request indoor recess or online schooling if advised.",
            "Do not open windows even for 'fresh air'—outdoor air is hazardous.",
            "Do not venture out during early morning or late night hours when particulate concentrations peak.",
            "Do not smoke or allow any form of indoor combustion."
        ],
        'vulnerable': "Extreme danger for pregnant women, infants, seniors, and cardiopulmonary patients. High risk of acute exacerbations.",
        'mask': "N95 / N99 mask mandatory for all individuals venturing outside.",
        'purifier': "HEPA filtration is essential indoors. Seal window drafts with damp cloths or sealing strips."
    },
    'Severe': {
        'color': '#881337',
        'text_color': '#ffffff',
        'range': '401 - 500+',
        'severity': 'Public Health Emergency & Hazardous Conditions',
        'icon': '☠️',
        'description': 'Affects healthy people and seriously impacts those with existing diseases. Public health emergency level.',
        'dos': [
            "STRICT CONFINEMENT INDOORS: Treat conditions as an environmental emergency.",
            "Seal all drafty windows, under-door gaps, and balcony ventilators with weather stripping or damp towels.",
            "Operate HEPA air purifiers around the clock at maximum capacity.",
            "Anyone venturing outside MUST wear an airtight N95/N99 respirator mask with zero air leakage.",
            "Keep emergency medical numbers, rescue inhalers, nebulizers, and oxygen supplies on standby.",
            "Wash eyes with clean cool water and rinse nasal passages with sterile saline rinse if irritated.",
            "Seek immediate emergency medical attention at the first sign of severe chest pain, dizziness, or hypoxia."
        ],
        'donts': [
            "DO NOT STEP OUTDOORS under any non-life-threatening circumstances.",
            "DO NOT allow children, elderly, or family pets to step outside onto balconies or open yards.",
            "DO NOT commute by open two-wheeler, bicycle, or open-window autorickshaws.",
            "DO NOT perform any strenuous physical chores even indoors; keep exertion to minimum.",
            "DO NOT operate gas stoves or room heaters without dedicated venting.",
            "DO NOT delay medical assistance if experiencing difficulty breathing."
        ],
        'vulnerable': "CRITICAL THREAT TO ALL HUMANS. Even healthy adults will experience respiratory tract irritation, headaches, and lung stress.",
        'mask': "N95 / N99 / FFP3 mask with tight facial seal is mandatory for any emergency outdoors.",
        'purifier': "Continuous multi-stage HEPA + activated carbon filtration required in all living quarters."
    }
}

# Indian NAAQS limits by averaging period; not instantaneous safety thresholds
NAAQS_LIMITS = {
    'PM2.5': {'limit': 60.0, 'period': '24-hour', 'unit': 'µg/m³', 'name': 'Fine Particulate Matter'},
    'PM10': {'limit': 100.0, 'period': '24-hour', 'unit': 'µg/m³', 'name': 'Coarse Particulates'},
    'NO2': {'limit': 80.0, 'period': '24-hour', 'unit': 'µg/m³', 'name': 'Nitrogen Dioxide'},
    'NH3': {'limit': 400.0, 'period': '24-hour', 'unit': 'µg/m³', 'name': 'Ammonia'},
    'SO2': {'limit': 80.0, 'period': '24-hour', 'unit': 'µg/m³', 'name': 'Sulphur Dioxide'},
    'CO': {'limit': 2.0, 'period': '8-hour', 'unit': 'mg/m³', 'name': 'Carbon Monoxide'},
    'O3': {'limit': 100.0, 'period': '8-hour', 'unit': 'µg/m³', 'name': 'Ozone'},
    'Benzene': {'limit': 5.0, 'period': 'Annual', 'unit': 'µg/m³', 'name': 'Benzene'}
}

# ---------------------------------------------------------
# Session State Initialization (Persistent Inputs)
# ---------------------------------------------------------
for feat, default_val in DEFAULT_VALUES.items():
    if feat not in st.session_state:
        st.session_state[feat] = float(default_val)

if "preset_feedback" not in st.session_state:
    st.session_state["preset_feedback"] = ""

# ---------------------------------------------------------
# Load Trained ML Models
# ---------------------------------------------------------
@st.cache_resource
def load_ml_models():
    reg_path = ARTIFACTS_DIR / "best_model_reg.pkl"
    clf_path = ARTIFACTS_DIR / "best_model_clf.pkl"
    
    with open(reg_path, "rb") as f:
        reg_model = pickle.load(f)
    with open(clf_path, "rb") as f:
        clf_model = pickle.load(f)
        
    return reg_model, clf_model

try:
    reg_model, clf_model = load_ml_models()
except Exception as e:
    st.error(f"Error loading trained models: {e}. Please ensure both model files are in the project's artifacts folder.")
    st.stop()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌿 Air Quality Index (AQI) Predictor & Health Advisory</h1>
    <p>Estimate AQI and AQI Category from 11 air pollutant readings using Machine Learning (Random Forest & XGBoost). Get customized, general informational air-quality precautions on what to do and what NOT to do to stay safe.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Presets and Information
# ---------------------------------------------------------
def apply_selected_preset():
    choice = st.session_state.get("selected_preset_dropdown")
    if choice and choice in PRESETS and PRESETS[choice] is not None:
        for f_name, f_val in PRESETS[choice].items():
            st.session_state[f_name] = float(f_val)
        st.session_state["preset_feedback"] = f"✅ Loaded preset scenario: {choice}"

with st.sidebar:
    st.header("⚡ Scenario Presets")
    st.caption("Quickly populate inputs with representative pollutant measurements:")
    
    st.selectbox(
        "Choose a Preset Scenario:",
        options=list(PRESETS.keys()),
        index=0,
        key="selected_preset_dropdown",
        on_change=apply_selected_preset
    )
    
    if st.button("📥 Apply Preset", use_container_width=True, help="Use the selected pollution scenario"):
        apply_selected_preset()
        st.rerun()
    if st.button("🔄 Reset Inputs", use_container_width=True, help="Restore the default pollutant measurements"):
        for f_name, f_val in DEFAULT_VALUES.items():
            st.session_state[f_name] = float(f_val)
        st.session_state["preset_feedback"] = "🔄 Reset all values to defaults."
        st.rerun()

    if st.session_state["preset_feedback"]:
        st.success(st.session_state["preset_feedback"])

    st.markdown("---")
    st.subheader("📊 AQI Category Scale (CPCB)")
    scale_html = """
    <div style='font-size: 0.9rem; line-height: 1.8;'>
        <div style='display:flex; justify-content:space-between;'><span style='color:#10B981; font-weight:bold;'>● Good</span><span>0 - 50</span></div>
        <div style='display:flex; justify-content:space-between;'><span style='color:#84CC16; font-weight:bold;'>● Satisfactory</span><span>51 - 100</span></div>
        <div style='display:flex; justify-content:space-between;'><span style='color:#F59E0B; font-weight:bold;'>● Moderate</span><span>101 - 200</span></div>
        <div style='display:flex; justify-content:space-between;'><span style='color:#F97316; font-weight:bold;'>● Poor</span><span>201 - 300</span></div>
        <div style='display:flex; justify-content:space-between;'><span style='color:#EF4444; font-weight:bold;'>● Very Poor</span><span>301 - 400</span></div>
        <div style='display:flex; justify-content:space-between;'><span style='color:#881337; font-weight:bold;'>● Severe</span><span>401 - 500+</span></div>
    </div>
    """
    st.markdown(scale_html, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("ℹ️ Model Architecture")
    st.caption("""
    - **Regression**: Random Forest Regressor (50 Estimators) predicts numerical AQI.
    - **Classification**: XGBoost Classifier predicts CPCB AQI Bucket.
    - **Dataset**: CPCB daily ambient air pollution monitoring across Indian cities & stations.
    """)

# Executive overview powered only by verified held-out regression metrics.
with (ANALYTICS_DIR / "regression_metrics.json").open(encoding="utf-8") as _metric_file:
    _summary_metrics = json.load(_metric_file)
_best = min(_summary_metrics["results"], key=lambda item: item["rmse"])
st.markdown("#### ✨ Model performance at a glance")
from html import escape as _escape_html
st.markdown(f"""
<div class="aqi-kpi-grid">
  <div class="aqi-kpi best"><div class="kpi-caption">🏆 Best regression model</div><div class="kpi-value">{_escape_html(str(_best['model']))}</div><div class="kpi-foot">Lowest test RMSE</div></div>
  <div class="aqi-kpi"><div class="kpi-caption">📈 Test R² score</div><div class="kpi-value">{_best['r2']:.4f}</div><div class="kpi-foot">Higher is better</div></div>
  <div class="aqi-kpi"><div class="kpi-caption">🎯 Test RMSE</div><div class="kpi-value">{_best['rmse']:.2f}</div><div class="kpi-foot">Lower is better · AQI units</div></div>
  <div class="aqi-kpi"><div class="kpi-caption">🧪 Models compared</div><div class="kpi-value">{len(_summary_metrics['results'])}</div><div class="kpi-foot">Same holdout dataset</div></div>
</div>
""", unsafe_allow_html=True)
st.caption("Metrics from the held-out date-grouped test set; not accuracy percentages or live air-quality readings.")

tab_prediction, tab_analytics, tab_guide = st.tabs(["🎯 AQI Prediction", "📊 Model Analytics", "🛡️ Guide & Standards"])
with tab_guide:
    st.subheader("Understanding the AQI dashboard")
    st.write("Enter 11 pollutant measurements in **AQI Prediction** and click Calculate to view the numerical AQI estimate, independently predicted category, safety precautions and a downloadable report.")
    st.info("The predictions are model estimates, not official live monitoring or medical advice. National standards use specific averaging periods; a single reading cannot establish compliance.")
    st.markdown("**AQI categories (CPCB):** Good (0–50), Satisfactory (51–100), Moderate (101–200), Poor (201–300), Very Poor (301–400), and Severe (401+).")
    st.markdown("**Models compared:** KNN, SVR with RBF kernel, Random Forest and XGBoost; Random Forest serves numerical predictions and a separate XGBoost classifier predicts categories.")

with tab_prediction:
    # ---------------------------------------------------------
    # Step 1: Input Pollutant Concentrations
    # ---------------------------------------------------------
    st.markdown("### 🧪 Step 1: Input Pollutant Concentrations")
    st.write("Adjust the pollutant values below. You can type numbers directly or use the increment buttons:")

    col_grp1, col_grp2 = st.columns(2)

    with col_grp1:
        st.markdown("#### 🔬 Particulate Matter (PM)")
        st.caption("Microscopic solid and liquid droplets suspended in the atmosphere.")
    
        c1, c2 = st.columns(2)
        with c1:
            pm25 = st.number_input(
                "PM2.5 (µg/m³)",
                min_value=0.0, max_value=1200.0,
                step=1.0,
                key="PM2.5",
                help="Fine inhalable particles with diameters 2.5 µm and smaller. NAAQS limit: 60 µg/m³."
            )
        with c2:
            pm10 = st.number_input(
                "PM10 (µg/m³)",
                min_value=0.0, max_value=1500.0,
                step=1.0,
                key="PM10",
                help="Inhalable particles with diameters 10 µm and smaller. NAAQS limit: 100 µg/m³."
            )

        st.markdown("#### ⚡ Inorganic Gases (CO, SO2, O3)")
        st.caption("Combustion byproducts, industrial emissions, and photochemical oxidants.")
    
        g1, g2, g3 = st.columns(3)
        with g1:
            co = st.number_input(
                "CO (mg/m³)",
                min_value=0.0, max_value=200.0,
                step=0.1,
                key="CO",
                help="Carbon Monoxide from incomplete combustion. NAAQS 8-hr limit: 2.0 mg/m³."
            )
        with g2:
            so2 = st.number_input(
                "SO2 (µg/m³)",
                min_value=0.0, max_value=300.0,
                step=0.5,
                key="SO2",
                help="Sulphur Dioxide from coal power plants & refineries. NAAQS limit: 80 µg/m³."
            )
        with g3:
            o3 = st.number_input(
                "O3 (µg/m³)",
                min_value=0.0, max_value=350.0,
                step=0.5,
                key="O3",
                help="Ground-level Ozone formed by sunlight + NOx + VOCs. NAAQS limit: 100 µg/m³."
            )

    with col_grp2:
        st.markdown("#### 🚗 Nitrogen Compounds & Ammonia")
        st.caption("Emissions predominantly from vehicular engines and agricultural sources.")
    
        n1, n2 = st.columns(2)
        with n1:
            no = st.number_input(
                "NO (µg/m³)",
                min_value=0.0, max_value=400.0,
                step=0.5,
                key="NO",
                help="Nitric Oxide emitted from high-temperature combustion engines."
            )
            nox = st.number_input(
                "NOx (ppb)",
                min_value=0.0, max_value=500.0,
                step=0.5,
                key="NOx",
                help="Total oxides of nitrogen (NO + NO2 combined concentration)."
            )
        with n2:
            no2 = st.number_input(
                "NO2 (µg/m³)",
                min_value=0.0, max_value=400.0,
                step=0.5,
                key="NO2",
                help="Nitrogen Dioxide, irritates bronchial passages. NAAQS limit: 80 µg/m³."
            )
            nh3 = st.number_input(
                "NH3 (µg/m³)",
                min_value=0.0, max_value=400.0,
                step=0.5,
                key="NH3",
                help="Ammonia from agricultural activities and livestock waste. NAAQS limit: 400 µg/m³."
            )

        st.markdown("#### 🧪 Volatile Organic Compounds (VOCs)")
        st.caption("Industrial solvents, petrochemical vapors, and evaporative emissions.")
    
        v1, v2 = st.columns(2)
        with v1:
            benzene = st.number_input(
                "Benzene (µg/m³)",
                min_value=0.0, max_value=500.0,
                step=0.1,
                key="Benzene",
                help="Carcinogenic aromatic hydrocarbon in vehicle exhaust. NAAQS limit: 5.0 µg/m³."
            )
        with v2:
            toluene = st.number_input(
                "Toluene (µg/m³)",
                min_value=0.0, max_value=500.0,
                step=0.1,
                key="Toluene",
                help="Aromatic hydrocarbon solvent found in paints, coatings, and fuel blends."
            )

    # ---------------------------------------------------------
    # Predict Trigger Button
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Calculate AQI, Bucket & Safety Precautions", use_container_width=True, type="primary")


with tab_analytics:
    # ---------------------------------------------------------
    # Model analytics accessible without running an individual prediction.
    # ---------------------------------------------------------
    with st.expander("📊 Four-Algorithm Model Comparison & Performance Analytics", expanded=False):
        analytics_dir = ANALYTICS_DIR
        metrics_path = analytics_dir / "regression_metrics.json"
        if metrics_path.exists():
            try:
                with metrics_path.open(encoding="utf-8") as source:
                    comparison = json.load(source)
                metrics_df = pd.DataFrame(comparison["results"])
                st.caption("Same date-grouped holdout for KNN, SVR (RBF), Random Forest and XGBoost. Smaller MAE/RMSE and larger R² are better.")
                st.dataframe(metrics_df.rename(columns={"model":"Model", "mae":"MAE", "mse":"MSE", "rmse":"RMSE", "r2":"R²"}), hide_index=True, use_container_width=True)
                st.success("Best test RMSE: " + comparison["best_model_by_rmse"])
                for picture in ("model_performance_comparison.png", "actual_vs_predicted_random_forest.png", "residuals_random_forest.png"):
                    chart_file = analytics_dir / picture
                    if chart_file.is_file():
                        st.image(chart_file, use_container_width=True)
                st.download_button("Download model metrics (CSV)", data=metrics_df.to_csv(index=False), file_name="aqi_four_models.csv", mime="text/csv")
                st.caption("This is one date-grouped random holdout, not a forecasting or external-validation benchmark.")
            except (KeyError, ValueError, OSError, TypeError) as exc:
                st.error(f"Model analytics could not be loaded: {exc}")
        else:
            st.warning("Missing analytics; run python model_comparison.py to regenerate.")


with tab_prediction:
    # Explicitly initiate predictions only on user request.
    if not predict_clicked:
        st.info("Enter pollutant readings or select a preset, then click **Calculate AQI, Bucket & Safety Precautions** to generate results and a downloadable report.")
        st.stop()

    # ---------------------------------------------------------
    # Model Inference Execution
    # ---------------------------------------------------------
    input_dict = {
        'PM2.5': float(st.session_state['PM2.5']),
        'PM10': float(st.session_state['PM10']),
        'NO': float(st.session_state['NO']),
        'NO2': float(st.session_state['NO2']),
        'NOx': float(st.session_state['NOx']),
        'NH3': float(st.session_state['NH3']),
        'CO': float(st.session_state['CO']),
        'SO2': float(st.session_state['SO2']),
        'O3': float(st.session_state['O3']),
        'Benzene': float(st.session_state['Benzene']),
        'Toluene': float(st.session_state['Toluene'])
    }

    input_df = pd.DataFrame([input_dict], columns=FEATURE_COLS)

    # Reject invalid measurements instead of sending them into a trained model.
    if not np.isfinite(input_df.to_numpy(dtype=float)).all() or (input_df < 0).any().any():
        st.error("Pollutant concentrations must be finite, non-negative numbers.")
        st.stop()

    # Predict AQI (Regression)
    pred_aqi_raw = reg_model.predict(input_df)[0]
    pred_aqi = max(0.0, float(pred_aqi_raw))  # AQI cannot be negative

    # Classification uses exactly the same 11 pollutant features as regression.
    # Do not pass numerical AQI (actual or predicted) into the classifier.
    clf_input_df = input_df.copy()

    # Predict AQI Bucket (Classification)
    pred_bucket_code = int(clf_model.predict(clf_input_df)[0])
    pred_bucket = BUCKET_MAPPING.get(pred_bucket_code, "Moderate")

    # Classifier Probabilities
    bucket_probs = clf_model.predict_proba(clf_input_df)[0]
    classes = clf_model.classes_

    cfg = BUCKET_CONFIG.get(pred_bucket, BUCKET_CONFIG['Moderate'])
    # Category classifiers and numerical regressors are independent models.
    def bucket_from_numerical_aqi(aqi):
        if aqi <= 50: return 'Good'
        if aqi <= 100: return 'Satisfactory'
        if aqi <= 200: return 'Moderate'
        if aqi <= 300: return 'Poor'
        if aqi <= 400: return 'Very Poor'
        return 'Severe'

    numeric_bucket = bucket_from_numerical_aqi(pred_aqi)
    if numeric_bucket != pred_bucket:
        st.warning(f"Model disagreement: numerical AQI {pred_aqi:.1f} corresponds to **{numeric_bucket}**, but the separate category classifier predicts **{pred_bucket}**. These are independent predictions; do not interpret the combined output as an official live AQI reading.")
    st.caption("Model estimates based on user-entered readings; not an official real-time monitor or medical diagnosis.")

    # ---------------------------------------------------------
    # Step 2: Prediction Results Dashboard
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Step 2: Prediction Results & Air Quality Status")

    # High-Contrast Banner
    summary_card_html = f"""
    <div class="aqi-contrast-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border: 3px solid {cfg['color']}; border-radius: 16px; padding: 1.6rem 2rem; margin-bottom: 1.5rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div>
                <span class="aqi-bucket-pill" style="background-color: {cfg['color']}; color: {cfg['text_color']}; font-weight: 800; font-size: 1.15rem; padding: 0.45rem 1.4rem; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.05em; display: inline-block;">
                    {cfg['icon']} {pred_bucket}
                </span>
                <h2 style="margin: 0.9rem 0 0.3rem 0; color: #ffffff !important; font-size: 2.2rem; font-weight: 800;">
                    Predicted AQI: {pred_aqi:.1f}
                </h2>
                <p style="margin: 0; color: #cbd5e1 !important; font-size: 1.1rem; font-weight: 500;">
                    <b>{cfg['severity']}</b> &bull; Standard Category Range: <b class="aqi-accent" style="color: {cfg['color']};">{cfg['range']}</b>
                </p>
            </div>
            <div style="text-align: right; background: rgba(255, 255, 255, 0.08); padding: 0.8rem 1.4rem; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Predicted Category Probability</div>
                <div class="aqi-accent" style="font-size: 2.1rem; font-weight: 800; color: {cfg['color']};">
                    {bucket_probs[pred_bucket_code] * 100:.1f}%
                </div>
                <div style="font-size: 0.78rem; color: #e2e8f0 !important; line-height: 1.4; margin-top: 0.4rem; max-width: 230px; text-align: right;">
                    Estimated probability of this AQI category, not the model's overall accuracy.
                </div>
            </div>
        </div>
        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.15); color: #e2e8f0 !important; font-size: 1.05rem; line-height: 1.6;">
            <b style="color: #ffffff;">Impact Summary:</b> {cfg['description']}
        </div>
    </div>
    """
    st.markdown(summary_card_html, unsafe_allow_html=True)

    # Visualizations: Speedometer Gauge & Probabilities
    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        st.subheader("⏱️ AQI Speedometer Gauge")
    
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_aqi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Air Quality Index (AQI)", 'font': {'size': 18, 'color': '#ffffff' if st.get_option("theme.base") == "dark" else '#1e293b'}},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': cfg['color'], 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 1,
                'bordercolor': "#64748b",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.45)'},
                    {'range': [50, 100], 'color': 'rgba(132, 204, 22, 0.45)'},
                    {'range': [100, 200], 'color': 'rgba(245, 158, 11, 0.45)'},
                    {'range': [200, 300], 'color': 'rgba(249, 115, 22, 0.45)'},
                    {'range': [300, 400], 'color': 'rgba(239, 68, 68, 0.45)'},
                    {'range': [400, 500], 'color': 'rgba(136, 19, 55, 0.55)'}
                ],
                'threshold': {
                    'line': {'color': '#ffffff', 'width': 4},
                    'thickness': 0.8,
                    'value': pred_aqi
                }
            }
        ))
        fig_gauge.update_layout(
            height=320,
            margin=dict(l=25, r=25, t=40, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with viz_col2:
        st.subheader("📊 Category Probability Distribution")
    
        prob_data = []
        for code in classes:
            b_name = BUCKET_MAPPING.get(code, f"Class {code}")
            prob_val = bucket_probs[code] * 100
            prob_data.append({
                'Category': b_name,
                'Probability (%)': prob_val,
                'Color': BUCKET_CONFIG.get(b_name, {}).get('color', '#64748b')
            })
    
        prob_df = pd.DataFrame(prob_data)
        fig_prob = px.bar(
            prob_df,
            x='Probability (%)',
            y='Category',
            orientation='h',
            text='Probability (%)',
            color='Category',
            color_discrete_map={row['Category']: row['Color'] for _, row in prob_df.iterrows()}
        )
        fig_prob.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside',
            showlegend=False
        )
        fig_prob.update_layout(
            height=320,
            margin=dict(l=10, r=40, t=20, b=20),
            xaxis=dict(range=[0, 105], title="Confidence (%)"),
            yaxis=dict(title=None, categoryorder='array', categoryarray=['Severe', 'Very Poor', 'Poor', 'Moderate', 'Satisfactory', 'Good']),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    # ---------------------------------------------------------
    # Step 3: Precautions & Safety Advisory (DOs & DON'Ts)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown(f"### 🛡️ Step 3: Health Precautions & Safety Advisory for <span style='color:{cfg['color']}; font-weight:800;'>{pred_bucket} Air Quality</span>", unsafe_allow_html=True)
    st.write("Follow these high-contrast medical precautions and preventive steps tailored to your predicted pollution levels:")

    col_dos, col_donts = st.columns(2)

    with col_dos:
        st.markdown("""
        <div class="aqi-contrast-card" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); color: #ffffff !important; border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(6, 95, 70, 0.3);">
            <h3 style="color: #ffffff !important; margin: 0; font-size: 1.3rem; font-weight: 800;">
                ✅ What You SHOULD Do (Recommended Actions)
            </h3>
            <p style="color: #d1fae5 !important; margin: 4px 0 0 0; font-size: 0.92rem;">Evidence-based protective routines to stay healthy</p>
        </div>
        """, unsafe_allow_html=True)
    
        for item in cfg['dos']:
            card_html = f"""
            <div class="aqi-contrast-card" style="background-color: #064e3b; color: #ffffff !important; border-left: 6px solid #10b981; padding: 14px 18px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
                <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #6ee7b7 !important; margin-bottom: 4px;">
                    ✔ ACTION REQUIRED
                </div>
                <div style="font-size: 1.02rem; color: #ffffff !important; line-height: 1.55; font-weight: 500;">
                    {item}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    with col_donts:
        st.markdown("""
        <div class="aqi-contrast-card" style="background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%); color: #ffffff !important; border-radius: 12px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; box-shadow: 0 4px 10px rgba(153, 27, 27, 0.3);">
            <h3 style="color: #ffffff !important; margin: 0; font-size: 1.3rem; font-weight: 800;">
                ❌ What You SHOULD NOT Do (Hazardous Mistakes)
            </h3>
            <p style="color: #fee2e2 !important; margin: 4px 0 0 0; font-size: 0.92rem;">Dangerous exposures and actions to strictly avoid</p>
        </div>
        """, unsafe_allow_html=True)
    
        for item in cfg['donts']:
            card_html = f"""
            <div class="aqi-contrast-card" style="background-color: #7f1d1d; color: #ffffff !important; border-left: 6px solid #ef4444; padding: 14px 18px; border-radius: 10px; margin-bottom: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
                <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #fca5a5 !important; margin-bottom: 4px;">
                    ✖ STRICTLY AVOID
                </div>
                <div style="font-size: 1.02rem; color: #ffffff !important; line-height: 1.55; font-weight: 500;">
                    {item}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Sensitive Groups, Masks & Air Purification
    # ---------------------------------------------------------
    st.markdown("#### 👥 Sensitive Groups, Masks & Indoor Purification")

    v_col1, v_col2, v_col3 = st.columns(3)

    with v_col1:
        vulnerable_card = f"""
        <div class="aqi-contrast-card" style="background-color: #78350f; color: #ffffff !important; border-left: 6px solid #f59e0b; padding: 1.2rem; border-radius: 12px; height: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
            <div style="font-size: 0.9rem; font-weight: 800; text-transform: uppercase; color: #fde68a !important; margin-bottom: 4px;">
                ⚠️ VULNERABLE POPULATIONS
            </div>
            <p style="font-size: 0.9rem; color: #fef3c7 !important; margin-bottom: 0.6rem;">
                <b>High Risk:</b> Infants, seniors, pregnant women, asthmatics, and heart patients.
            </p>
            <div style="font-size: 0.98rem; color: #ffffff !important; line-height: 1.5;">
                {cfg['vulnerable']}
            </div>
        </div>
        """
        st.markdown(vulnerable_card, unsafe_allow_html=True)

    with v_col2:
        mask_card = f"""
        <div class="aqi-contrast-card" style="background-color: #0c4a6e; color: #ffffff !important; border-left: 6px solid #0284c7; padding: 1.2rem; border-radius: 12px; height: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
            <div style="font-size: 0.9rem; font-weight: 800; text-transform: uppercase; color: #bae6fd !important; margin-bottom: 4px;">
                😷 RESPIRATORY MASK GUIDANCE
            </div>
            <p style="font-size: 0.9rem; color: #e0f2fe !important; margin-bottom: 0.6rem;">
                <b>Mask Protocol:</b> Effective filtration against ambient particulates.
            </p>
            <div style="font-size: 0.98rem; color: #ffffff !important; line-height: 1.5;">
                {cfg['mask']}
            </div>
        </div>
        """
        st.markdown(mask_card, unsafe_allow_html=True)

    with v_col3:
        purifier_card = f"""
        <div class="aqi-contrast-card" style="background-color: #1e3a8a; color: #ffffff !important; border-left: 6px solid #6366f1; padding: 1.2rem; border-radius: 12px; height: 100%; box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
            <div style="font-size: 0.9rem; font-weight: 800; text-transform: uppercase; color: #c7d2fe !important; margin-bottom: 4px;">
                🏠 INDOOR AIR & PURIFIERS
            </div>
            <p style="font-size: 0.9rem; color: #e0e7ff !important; margin-bottom: 0.6rem;">
                <b>Indoor Strategy:</b> Air filtration and room sealing.
            </p>
            <div style="font-size: 0.98rem; color: #ffffff !important; line-height: 1.5;">
                {cfg['purifier']}
            </div>
        </div>
        """
        st.markdown(purifier_card, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Step 4: Pollutants vs. National Standards (NAAQS)
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📈 Step 4: Measured Pollutants vs. National Standards (NAAQS)")
    st.write("Reference comparison with Indian NAAQS limits. Each pollutant has its own averaging period (24-hour, 8-hour or annual):")

    comparison_rows = []
    for p, meta in NAAQS_LIMITS.items():
        if p in input_dict:
            curr_val = input_dict[p]
            safe_lim = meta['limit']
            ratio = (curr_val / safe_lim) * 100
            status = "Above reference ⚠️" if curr_val > safe_lim else "At/below reference ✅"
            comparison_rows.append({
                'Pollutant': p,
                'Full Name': meta['name'],
                'Measured Value': curr_val,
                'NAAQS Limit': safe_lim,
                'Averaging Period': meta['period'],
                'Unit': meta['unit'],
                '% of Safe Limit': round(ratio, 1),
                'Status': status
            })

    comp_df = pd.DataFrame(comparison_rows)

    col_tbl, col_chart = st.columns([1.1, 0.9])

    with col_tbl:
        st.dataframe(
            comp_df[['Pollutant', 'Full Name', 'Measured Value', 'NAAQS Limit', 'Averaging Period', 'Unit', '% of Safe Limit', 'Status']],
            use_container_width=True,
            hide_index=True
        )

    with col_chart:
        fig_comp = px.bar(
            comp_df,
            x='% of Safe Limit',
            y='Pollutant',
            orientation='h',
            color='% of Safe Limit',
            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444', '#881337'],
            title="Pollutant Reading as % of NAAQS Reference"
        )
        fig_comp.add_vline(x=100, line_dash="dash", line_color="#dc2626", annotation_text="100% reference", annotation_position="top right")
        fig_comp.update_layout(
            height=330,
            margin=dict(l=10, r=20, t=40, b=20),
            xaxis=dict(title="% of NAAQS reference value"),
            yaxis=dict(title=None),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    st.caption("Important: A single entered measurement is not necessarily the 8-hour, 24-hour or annual average required for compliance assessment. This chart is illustrative, not an official NAAQS compliance verdict.")

    # ---------------------------------------------------------
    # Step 5: Export / Download Report
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📥 Step 5: Download Personalized Safety Report")

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_content = f"""=====================================================
    AIR QUALITY INDEX (AQI) ASSESSMENT & SAFETY ADVISORY
    Generated on: {now_str}
    =====================================================

    1. PREDICTION RESULTS
    -----------------------------------------------------
    - Predicted Numerical AQI: {pred_aqi:.2f}
    - Air Quality Category:   {pred_bucket.upper()} (Scale: {cfg['range']})
    - Overall Health Risk:    {cfg['severity']}
    - Predicted Category Probability: {bucket_probs[pred_bucket_code] * 100:.1f}% (model estimate; not overall accuracy)

    2. MEASURED POLLUTANT READINGS
    -----------------------------------------------------
    - PM2.5:     {st.session_state['PM2.5']:.2f} µg/m³ (NAAQS reference: 60.0 µg/m³)
    - PM10:      {st.session_state['PM10']:.2f} µg/m³ (NAAQS reference: 100.0 µg/m³)
    - NO:        {st.session_state['NO']:.2f} µg/m³
    - NO2:       {st.session_state['NO2']:.2f} µg/m³ (NAAQS reference: 80.0 µg/m³)
    - NOx:       {st.session_state['NOx']:.2f} ppb
    - NH3:       {st.session_state['NH3']:.2f} µg/m³ (NAAQS reference: 400.0 µg/m³)
    - CO:        {st.session_state['CO']:.2f} mg/m³ (NAAQS reference: 2.0 mg/m³)
    - SO2:       {st.session_state['SO2']:.2f} µg/m³ (NAAQS reference: 80.0 µg/m³)
    - O3:        {st.session_state['O3']:.2f} µg/m³ (NAAQS reference: 100.0 µg/m³)
    - Benzene:   {st.session_state['Benzene']:.2f} µg/m³ (NAAQS reference: 5.0 µg/m³)
    - Toluene:   {st.session_state['Toluene']:.2f} µg/m³

    3. RECOMMENDED ACTIONS (WHAT TO DO)
    -----------------------------------------------------
    """
    for idx, do in enumerate(cfg['dos'], 1):
        report_content += f"{idx}. {do}\n"

    report_content += """
    4. ACTIONS TO AVOID (WHAT NOT TO DO)
    -----------------------------------------------------
    """
    for idx, dont in enumerate(cfg['donts'], 1):
        report_content += f"{idx}. {dont}\n"

    report_content += f"""
    5. SPECIAL ADVISORIES
    -----------------------------------------------------
    - Vulnerable Groups:    {cfg['vulnerable']}
    - Mask Requirement:     {cfg['mask']}
    - Indoor Air Purifier:  {cfg['purifier']}

    =====================================================
    Health advisory compiled based on Central Pollution Control Board (CPCB)
    and WHO Ambient Air Quality Guidelines.
    =====================================================
    """

    st.download_button(
        label="📄 Download Air Quality & Health Advisory Report (.txt)",
        data=report_content,
        file_name=f"AQI_Report_{pred_bucket}_{int(pred_aqi)}.txt",
        mime="text/plain",
        use_container_width=True
    )


st.markdown("<br><hr><center style='color:#94a3b8; font-size:0.85rem;'>Air Quality Index Project &bull; Machine Learning Assisted Environmental Decision Support System</center>", unsafe_allow_html=True)
