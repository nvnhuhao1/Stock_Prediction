# (Imports and setup unchanged)

# Data & Model Configuration
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer.shap',
    "pipeline": 'finalized_model.tar.gz',
    "keys": ["INTL", "RELY", "DEXJPUS", "DEXCHUS", "SP500", "NASDAQCOM", "VIXCLS"],  # Fixed: Remove target
    "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ["INTL", "RELY", "DEXJPUS", "DEXCHUS", "SP500", "NASDAQCOM", "VIXCLS"]]  # Fixed: Remove target
}

# (load_pipeline and load_shap_explainer unchanged)

# Prediction Logic (unchanged)

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name), os.path.join(tempfile.gettempdir(), explainer_name))
    shap_values = explainer(input_df)
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[-1], max_display=10)  # Fixed: Use [-1] for last (new) row
    st.pyplot(fig)
    top_feature = shap_values[-1].feature_names[0]  # Fixed: Use [-1]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI (unchanged up to form)

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]  # Now 7 values
    # Prepare data
    base_df = df_features.copy()  # Safer: Use copy
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
    
    # Debug (add this temporarily to check shapes)
    print("base_df shape:", base_df.shape)
    print("data_row length:", len(data_row))
    print("input_df shape:", input_df.shape)
    
    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
