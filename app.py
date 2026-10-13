import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="AI Side Effect Extractor", layout="wide")
st.title("💊 AI-Powered Side Effect Extractor")
st.write("Upload case reports CSV or paste text to extract Drug, Side Effect, Severity")

# Simple extraction without heavy models
def extract_info(text):
    # Simple regex to find data from our generated format
    drug_match = re.search(r'prescribed (.+?) for', text, re.IGNORECASE)
    se_match = re.search(r'developed (.+?)\. Severity', text, re.IGNORECASE)
    sev_match = re.search(r'Severity was (.+?)\.', text, re.IGNORECASE)
    
    drug = drug_match.group(1) if drug_match else "Not found"
    se = se_match.group(1) if se_match else "Not found"
    sev = sev_match.group(1) if sev_match else "Not found"
    return drug, se, sev

uploaded = st.file_uploader("Upload case_reports_100.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} reports")
    
    results = []
    for _, row in df.iterrows():
        drug, se, sev = extract_info(row['text'])
        results.append({"report_id": row['report_id'], "drug": drug, "side_effect": se, "severity": sev, "original_text": row['text']})
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    st.bar_chart(results_df['side_effect'].value_counts())
    st.bar_chart(results_df['severity'].value_counts())
    
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results CSV", csv, "extracted_results.csv", "text/csv")

st.divider()
st.subheader("Test with single text")
test_text = st.text_area("Paste a case report:", "Case 101: A 45-year-old M patient was prescribed Metformin 500mg for type 2 diabetes. After 5 days, patient developed severe nausea and vomiting. Severity was moderate.")

if st.button("Extract"):
    d, s, sv = extract_info(test_text)
    st.json({"Drug": d, "Side Effect": s, "Severity": sv})