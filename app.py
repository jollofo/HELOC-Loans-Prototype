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
    }}
    .negative {{
        background-color: {SOFT_BUTTER};
        color: #7d6608;
        border-left: 10px solid {ACCENT_GOLD};
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
    base_path = os.path.join(script_dir, "model", "model_files")
    
    model = joblib.load(os.path.join(base_path, "heloc_random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    features = joblib.load(os.path.join(base_path, "feature_order.pkl"))
    info = joblib.load(os.path.join(base_path, "preprocessing_info.pkl"))
    config = joblib.load(os.path.join(base_path, "model_config.pkl"))
    return model, scaler, features, info, config

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

try:
    model, scaler, feature_names, info, config = load_model_artifacts()
    medians = info['median_values']
    threshold = config.get('threshold', 0.55)
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# Feature metadata for the UI
feature_meta = {
    'ExternalRiskEstimate': {'min': 0, 'max': 100, 'label': 'External Risk Estimate (High = Better)'},
    'MSinceOldestTradeOpen': {'min': 0, 'max': 1000, 'label': 'Months Since Oldest Trade Open'},
    'MSinceMostRecentTradeOpen': {'min': 0, 'max': 1000, 'label': 'Months Since Most Recent Trade Open'},
    'AverageMInFile': {'min': 0, 'max': 1000, 'label': 'Average Months in File'},
    'NumSatisfactoryTrades': {'min': 0, 'max': 100, 'label': 'Number of Satisfactory Trades'},
    'NumTrades60Ever2DerogPubRec': {'min': 0, 'max': 20, 'label': 'Trades 60+ Days Delinquent'},
    'NumTrades90Ever2DerogPubRec': {'min': 0, 'max': 20, 'label': 'Trades 90+ Days Delinquent'},
    'PercentTradesNeverDelq': {'min': 0, 'max': 100, 'label': 'Percent Trades Never Delinquent'},
    'MaxDelq2PublicRecLast12M': {'min': 0, 'max': 10, 'label': 'Max Delinquency (Last 12M)'},
    'MaxDelqEver': {'min': 0, 'max': 10, 'label': 'Max Delinquency Ever'},
    'NumTotalTrades': {'min': 0, 'max': 200, 'label': 'Total Number of Trades'},
    'NumTradesOpeninLast12M': {'min': 0, 'max': 20, 'label': 'Trades Opened in Last 12M'},
    'PercentInstallTrades': {'min': 0, 'max': 100, 'label': 'Percent Installment Trades'},
    'MSinceMostRecentInqexcl7days': {'min': 0, 'max': 50, 'label': 'Months Since Recent Inquiry'},
    'NumInqLast6M': {'min': 0, 'max': 100, 'label': 'Number of Inquiries (6M)'},
    'NumInqLast6Mexcl7days': {'min': 0, 'max': 100, 'label': 'Number of Inquiries (6M, excl. 7d)'},
    'NetFractionRevolvingBurden': {'min': 0, 'max': 300, 'label': 'Revolving Burden (%)'},
    'NetFractionInstallBurden': {'min': 0, 'max': 500, 'label': 'Installment Burden (%)'},
    'NumRevolvingTradesWBalance': {'min': 0, 'max': 50, 'label': 'Revolving Trades w/ Balance'},
    'NumInstallTradesWBalance': {'min': 0, 'max': 30, 'label': 'Installment Trades w/ Balance'},
    'NumBank2NatlTradesWHighUtilization': {'min': 0, 'max': 20, 'label': 'High Utilization Trades'},
    'PercentTradesWBalance': {'min': 0, 'max': 100, 'label': 'Percent Trades w/ Balance'}
}

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
    # Group inputs into logical categories
    with st.expander("Risk & History", expanded=True):
        for feat in ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'AverageMInFile', 'PercentTradesNeverDelq', 'MaxDelqEver']:
            meta = feature_meta[feat]
            input_data[feat] = st.number_input(meta['label'], min_value=float(meta['min']), max_value=float(meta['max']), value=float(medians.get(feat, meta['min'])))

    with st.expander("Trades Information"):
        for feat in ['NumSatisfactoryTrades', 'NumTotalTrades', 'PercentInstallTrades', 'PercentTradesWBalance']:
            meta = feature_meta[feat]
            input_data[feat] = st.number_input(meta['label'], min_value=float(meta['min']), max_value=float(meta['max']), value=float(medians.get(feat, meta['min'])))

    with st.expander("Delinquency & Inquiries"):
        for feat in ['NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'MaxDelq2PublicRecLast12M', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M']:
             meta = feature_meta[feat]
             input_data[feat] = st.number_input(meta['label'], min_value=float(meta['min']), max_value=float(meta['max']), value=float(medians.get(feat, meta['min'])))

    with st.expander("Burden & Utilization"):
        for feat in ['NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization']:
             meta = feature_meta.get(feat, {'min': 0, 'max': 100, 'label': feat})
             input_data[feat] = st.number_input(meta['label'], min_value=float(meta['min']), max_value=float(meta['max']), value=float(medians.get(feat, meta['min'])))

    # Handle any missing ones just in case
    for feat in feature_names:
        if feat not in input_data:
            input_data[feat] = medians.get(feat, 0.0)

    predict_btn = st.button("Evaluate Eligibility")

# Main Content Area
if predict_btn or 'prediction_done' in st.session_state:
    st.session_state['prediction_done'] = True
    
    # Prepare input dataframe
    input_df = pd.DataFrame([input_data])[feature_names]
    
    # Scale inputs
    X_scaled = scaler.transform(input_df)
    
    # Make prediction
    # Probability of being "Bad" (class 1)
    proba = model.predict_proba(X_scaled)[0, 1]
    
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
    
    # Calculate SHAP values
    explainer = get_shap_explainer(model)
    shap_values = explainer.shap_values(X_scaled)
    
    # Handle SHAP output format
    if isinstance(shap_values, list):
        # Class 1 is "Bad"
        local_shap_bad = shap_values[1][0]
    else:
        if len(shap_values.shape) == 3: # (num_samples, num_features, num_classes)
            local_shap_bad = shap_values[0, :, 1]
        else:
            local_shap_bad = shap_values[0][0:len(feature_names)]

    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_df.iloc[0].values,
        'SHAP_Impact': local_shap_bad
    }).sort_values('SHAP_Impact', key=abs, ascending=False)

    with col_exp1:
        st.subheader("Decision Factors (SHAP)")
        
        # Take top 10 most impactful features for the plot
        top_10 = explanation_df.head(10).copy()
        top_10['Impact_Pct'] = top_10['SHAP_Impact'] * 100
        # Use full labels from feature_meta
        top_10['Label'] = top_10.apply(lambda row: f"{feature_meta.get(row['Feature'], {'label': row['Feature']})['label']}", axis=1)
        top_10 = top_10.sort_values('Impact_Pct', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('#1e1e1e')
        
        colors = ['#ff4b4b' if x > 0 else '#00ff00' for x in top_10['Impact_Pct']]
        ax.barh(range(len(top_10)), top_10['Impact_Pct'], color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['Label'], color=ACCENT_GOLD, fontsize=10)
        ax.set_xlabel('Risk Contribution (%)', color=ACCENT_GOLD, fontsize=12)
        ax.set_title(f'Risk Impact (Total: {proba:.0%})', color=PRIMARY_YELLOW, fontsize=14, fontweight='bold')
        
        for spine in ax.spines.values():
            spine.set_color(ACCENT_GOLD)
        ax.tick_params(colors=ACCENT_GOLD)
        
        ax.axvline(x=0, color='white', linewidth=1, linestyle='-', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    with col_exp2:
        st.subheader("Improvement Suggestions")
        if is_denied:
            st.write("Focusing on these areas would most effectively lower the predicted risk profile:")
            
            # Suggest based on top 3 positive SHAP values (factors that increased risk)
            positive_impacts = explanation_df[explanation_df['SHAP_Impact'] > 0].head(3)
            
            if not positive_impacts.empty:
                for _, row in positive_impacts.iterrows():
                    feat = row['Feature']
                    label = feature_meta.get(feat, {'label': feat})['label']
                    
                    if 'Burden' in feat or 'Inq' in feat or 'Delq' in feat:
                        st.info(f"**{label}**: Decreasing this factor would improve eligibility.")
                    else:
                        st.info(f"**{label}**: Increasing this factor would improve eligibility.")
            else:
                st.info("Improve overall credit health by optimizing the factors shown in the chart.")
        else:
            st.success("The application demonstrates strong credit indicators.")
            st.write("Key strengths contributing to your approval:")
            # Show factors with negative SHAP (decreased risk)
            negative_impacts = explanation_df[explanation_df['SHAP_Impact'] < 0].head(3)
            for _, row in negative_impacts.iterrows():
                label = feature_meta.get(row['Feature'], {'label': row['Feature']})['label']
                st.write(f"✓ Strong performance in **{label}**")

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
