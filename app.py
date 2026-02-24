import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set page configuration
st.set_page_config(
    page_title="HELOC Decision Support System",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# URochester Color Palette
PRIMARY_NAVY = "#001E5F"
PRIMARY_YELLOW = "#FFD82B"
SECONDARY_BLUE = "#021BC3"
ACCENT_BLUE = "#66A2FF"
ACCENT_GOLD = "#FFC200"
SOFT_SKY = "#B7D3FF"
SOFT_BUTTER = "#FFF3B1"

# Custom CSS
st.markdown(f"""
    <style>
    .main {{
        background-color: #ffffff;
    }}
    .stApp {{
        background-color: black;
        color: {ACCENT_GOLD};
    }}
    /* Header styling */
    h1, h2, h3 {{
        color: {PRIMARY_YELLOW} !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    /* Button styling */
    .stButton>button {{
        width: 100%;
        border-radius: 4px;
        height: 3em;
        background-color: transparent;
        color: {PRIMARY_YELLOW} !important;
        font-weight: bold;
        border: 1px solid {PRIMARY_NAVY};
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {SOFT_BUTTER};
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_YELLOW};
        color: {PRIMARY_NAVY};
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{
        color: {PRIMARY_NAVY} !important;
    }}
    [data-testid="stSidebar"] summary {{
        background-color: {SOFT_BUTTER};
    }}
    /* Prediction boxes */
    .prediction-box {{
        padding: 25px;  
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    .positive {{
        background-color: {SOFT_SKY};
        color: {PRIMARY_NAVY};
        border-left: 10px solid {ACCENT_BLUE};
        h2 {{
            color: {PRIMARY_NAVY} !important;
        }}
    }}
    .negative {{
        background-color: {SOFT_BUTTER};
        color: #7d6608;
        border-left: 10px solid {ACCENT_GOLD};
        h2 {{
            color: {PRIMARY_NAVY} !important;
        }}
    }}
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: transparent !important;
        border: 1px solid {PRIMARY_NAVY} !important;
        color: {PRIMARY_NAVY} !important;
        border-radius: 4px;
        margin-bottom: 5px;
    }}
    .streamlit-expanderContent {{
        background-color: rgba(255, 255, 255, 0.05) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_model_artifacts():
    # Detect path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "model")
    
    model = joblib.load(os.path.join(base_path, "heloc_random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    features = joblib.load(os.path.join(base_path, "feature_order.pkl"))
    info = joblib.load(os.path.join(base_path, "preprocessing_info.pkl"))
    config = joblib.load(os.path.join(base_path, "model_config.pkl"))
    return model, scaler, features, info, config

@st.cache_resource
def get_shap_explainer(_model):
    # 'interventional' avoids C-level heap corruption seen on SHAP 0.46+ / Python 3.13
    return shap.TreeExplainer(_model, feature_perturbation='interventional')

try:
    model, scaler, feature_names, info, config = load_model_artifacts()
    medians = info['median_values']
    threshold = config.get('threshold', 0.55)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Feature metadata for the UI
# Exactly the 20 features the model was trained on, in model order
feature_meta = {
    'ExternalRiskEstimate':              {'min': 0,   'max': 100,  'label': 'External Risk Estimate (High = Better)'},
    'MSinceOldestTradeOpen':             {'min': 0,   'max': 1000, 'label': 'Months Since Oldest Trade Open'},
    'MSinceMostRecentTradeOpen':         {'min': -9,  'max': 1000, 'label': 'Months Since Most Recent Trade Open'},
    'AverageMInFile':                    {'min': 0,   'max': 1000, 'label': 'Average Months in File'},
    'NumSatisfactoryTrades':             {'min': 0,   'max': 100,  'label': 'Number of Satisfactory Trades'},
    'NumTrades60Ever2DerogPubRec':       {'min': 0,   'max': 20,   'label': 'Trades 60+ Days Delinquent Ever'},
    'PercentTradesNeverDelq':            {'min': 0,   'max': 100,  'label': 'Percent Trades Never Delinquent'},
    'MSinceMostRecentDelq':              {'min': -9,  'max': 1000, 'label': 'Months Since Most Recent Delinquency'},
    'MaxDelq2PublicRecLast12M':          {'min': 0,   'max': 10,   'label': 'Max Delinquency (Last 12M)'},
    'MaxDelqEver':                       {'min': 0,   'max': 10,   'label': 'Max Delinquency Ever'},
    'NumTradesOpeninLast12M':            {'min': -9,  'max': 20,   'label': 'Trades Opened in Last 12M'},
    'PercentInstallTrades':              {'min': 0,   'max': 100,  'label': 'Percent Installment Trades'},
    'MSinceMostRecentInqexcl7days':      {'min': -9,  'max': 50,   'label': 'Months Since Most Recent Inquiry'},
    'NumInqLast6M':                      {'min': 0,   'max': 100,  'label': 'Number of Inquiries (Last 6M)'},
    'NetFractionRevolvingBurden':        {'min': 0,   'max': 300,  'label': 'Net Fraction Revolving Burden (%)'},
    'NetFractionInstallBurden':          {'min': 0,   'max': 500,  'label': 'Net Fraction Installment Burden (%)'},
    'NumRevolvingTradesWBalance':        {'min': 0,   'max': 50,   'label': 'Revolving Trades with Balance'},
    'NumInstallTradesWBalance':          {'min': 0,   'max': 30,   'label': 'Installment Trades with Balance'},
    'NumBank2NatlTradesWHighUtilization':{'min': 0,   'max': 20,   'label': 'High Utilization Bank/National Trades'},
    'PercentTradesWBalance':             {'min': 0,   'max': 100,  'label': 'Percent Trades with Balance'},
}

# Helper: render a number input; sentinel (-9) features get an inline N/A checkbox
def render_input(feat):
    meta = feature_meta[feat]
    is_sentinel = meta['min'] == -9
    if is_sentinel:
        col_label, col_check = st.columns([3, 1])
        with col_check:
            na_checked = st.checkbox("N/A", key=f"{feat}_na",
                                     value=(medians.get(feat, 0) < 0))
        with col_label:
            if na_checked:
                st.number_input(meta['label'], value=-9.0, disabled=True,
                                key=f"{feat}_disabled")
                return -9.0
            else:
                return st.number_input(
                    meta['label'],
                    min_value=0.0,
                    max_value=float(meta['max']),
                    value=float(max(medians.get(feat, 0), 0)),
                    key=f"{feat}_input"
                )
    else:
        return st.number_input(
            meta['label'],
            min_value=float(meta['min']),
            max_value=float(meta['max']),
            value=float(medians.get(feat, meta['min'])),
            key=f"{feat}_input"
        )

# App Layout
st.title("Simon Bank of Rochester")
st.header("HELOC Eligibility Decision Support System")
st.markdown("---")

# Sidebar for application input
st.sidebar.header("Applicant Data")
input_data = {}

# Create columns for the form
col1, col2 = st.columns([2, 1])

with st.sidebar:
    st.write("Enter the applicant's credit details:")

    # Render every field in the exact order the model expects
    for feat in feature_names:
        input_data[feat] = render_input(feat)

    predict_btn = st.button("Evaluate Eligibility")



# Main Content Area
if predict_btn or 'prediction_done' in st.session_state:
    st.session_state['prediction_done'] = True
    
    # Prepare input dataframe (Random Forest is scale-invariant; raw values are correct)
    input_df = pd.DataFrame([input_data])[feature_names]

    # Make prediction — P(Bad) = class index 1
    proba = model.predict_proba(input_df)[0, 1]
    
    is_denied = proba >= threshold
    
    st.subheader("Decision Analysis")
    
    if is_denied:
        st.markdown(f"""
            <div class="prediction-box negative">
                <h2 style='margin:0'>Result: Application Denied</h2>
                <p style='font-size:1.2em'>Risk Priority Score: <b>{proba:.1%}</b> (Threshold: {threshold:.1%})</p>
                <p>Status: Application closed. Regulatory explanation required.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box positive">
                <h2 style='margin:0'>Result: Application Approved</h2>
                <p style='font-size:1.2em'>Risk Priority Score: <b>{proba:.1%}</b> (Threshold: {threshold:.1%})</p>
                <p>Status: Application forwarded to a Loan Officer for detailed manual assessment.</p>
            </div>
        """, unsafe_allow_html=True)

    # Explanation Section (SHAP)
    st.markdown("---")
    col_exp1, col_exp2 = st.columns(2)
    
    # Calculate SHAP values — pass numpy array to avoid C-ext issues with DataFrames
    explainer = get_shap_explainer(model)
    X_shap = input_df.values  # shape (1, n_features)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)

    # Handle all output shapes SHAP may return across versions:
    #   list  → [class_0_arr, class_1_arr], each (n_samples, n_features)
    #   3D arr → (n_samples, n_features, n_classes)
    #   2D arr → (n_samples, n_features)  [binary shorthand]
    if isinstance(shap_values, list):
        # Classic format — index 1 = Bad class
        shap_values_bad = np.array(shap_values[1][0]).flatten()
    else:
        sv = np.array(shap_values)
        if sv.ndim == 3:          # (n_samples, n_features, n_classes)
            shap_values_bad = sv[0, :, 1].flatten()
        else:                      # (n_samples, n_features)
            shap_values_bad = sv[0].flatten()

    # Flatten and trim to guaranteed equal length (defensive)
    shap_values_bad   = np.array(shap_values_bad).flatten()
    applicant_values  = np.array(input_df.iloc[0].values).flatten()
    feat_names_arr    = np.array(feature_names)
    min_len = min(len(feat_names_arr), len(applicant_values), len(shap_values_bad))
    feat_names_arr   = feat_names_arr[:min_len]
    applicant_values = applicant_values[:min_len]
    shap_values_bad  = shap_values_bad[:min_len]

    # Build explanation dataframe sorted by absolute impact
    explanation_df = pd.DataFrame({
        'Feature':     feat_names_arr,
        'Your_Value':  applicant_values,
        'SHAP_Impact': shap_values_bad
    }).sort_values('SHAP_Impact', key=abs, ascending=False)

    with col_exp1:
        st.subheader("Decision Factors (SHAP)")

        # Top 10 by absolute impact
        top_10 = explanation_df.head(10).copy()
        top_10['Impact_Pct'] = top_10['SHAP_Impact'] * 100
        # Label = feature name + raw applicant value (matches notebook)
        top_10['Label'] = top_10.apply(
            lambda row: f"{row['Feature']}: {row['Your_Value']:.0f}", axis=1
        )
        top_10 = top_10.sort_values('Impact_Pct', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('#1e1e1e')

        colors = ['#ff4b4b' if x > 0 else '#00c853' for x in top_10['Impact_Pct']]
        bars = ax.barh(range(len(top_10)), top_10['Impact_Pct'], color=colors, alpha=0.8)

        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['Label'], color=ACCENT_GOLD, fontsize=10)
        ax.set_xlabel('Risk Contribution (%)', color=ACCENT_GOLD, fontsize=12, fontweight='bold')
        ax.set_title(f'Feature Contributions to {proba:.0%} Default Risk',
                     color=PRIMARY_YELLOW, fontsize=14, fontweight='bold', pad=20)

        for spine in ax.spines.values():
            spine.set_color(ACCENT_GOLD)
        ax.tick_params(colors=ACCENT_GOLD)
        ax.axvline(x=0, color='white', linewidth=1, linestyle='-', alpha=0.3)

        # Percentage labels on bars (matches notebook)
        for i, (bar, val) in enumerate(zip(bars, top_10['Impact_Pct'])):
            if abs(val) > 1:
                label_x = val + (0.15 if val > 0 else -0.15)
                ax.text(label_x, i, f'{val:+.1f}%',
                        va='center', ha='left' if val > 0 else 'right',
                        fontsize=9, fontweight='bold', color=ACCENT_GOLD)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff4b4b', alpha=0.8, label='Increases Risk'),
            Patch(facecolor='#00c853', alpha=0.8, label='Decreases Risk'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10,
                  facecolor='#1e1e1e', labelcolor=ACCENT_GOLD, edgecolor=ACCENT_GOLD)
        ax.grid(axis='x', alpha=0.3, linestyle='--', color=ACCENT_GOLD)
        ax.set_axisbelow(True)

        plt.tight_layout()
        st.pyplot(fig)

    with col_exp2:
        st.subheader("Improvement Suggestions")
        if is_denied:
            st.write("Focusing on these areas would most effectively lower the predicted risk profile:")

            # Top 5 factors that increased risk (positive SHAP = more Bad)
            top_5_denial = explanation_df[explanation_df['SHAP_Impact'] > 0].head(5)
            if not top_5_denial.empty:
                for _, row in top_5_denial.iterrows():
                    feat  = row['Feature']
                    label = feature_meta.get(feat, {'label': feat})['label']
                    impact_pct = row['SHAP_Impact'] * 100
                    if 'Burden' in feat or 'Inq' in feat or 'Delq' in feat or 'Derog' in feat:
                        st.info(f"🔴 **{label}** (+{impact_pct:.1f}%): Reducing this would lower risk.")
                    else:
                        st.info(f"🔴 **{label}** (+{impact_pct:.1f}%): Improving this would lower risk.")
            else:
                st.info("Improve overall credit health by optimizing the factors shown in the chart.")
        else:
            st.success("The application demonstrates strong credit indicators.")
            st.write("Key strengths contributing to approval:")
            negative_impacts = explanation_df[explanation_df['SHAP_Impact'] < 0].head(5)
            for _, row in negative_impacts.iterrows():
                label = feature_meta.get(row['Feature'], {'label': row['Feature']})['label']
                impact_pct = abs(row['SHAP_Impact'] * 100)
                st.write(f"🟢 Strong performance in **{label}** (−{impact_pct:.1f}% risk)")

else:
    # Welcome Screen
    st.info("Enter applicant data in the sidebar and click 'Evaluate Eligibility' to begin.")
    
    st.write("""
    This Decision Support System (DSS) uses a Random Forest classifier trained on the FICO HELOC dataset.
    It evaluates creditworthiness based on 22 key financial indicators to assist loan officers in the initial screening process.
    """)
    
    # Show importance plot
    # st.subheader("Global Feature Importance")
    # importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(10)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # sns.barplot(x=importances.values, y=importances.index, color=PRIMARY_NAVY, ax=ax)
    # st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("© 2026 Simon Bank of Rochester - For Internal Use Only. This is an automated screening tool; final decisions rest with the primary loan officer.")
