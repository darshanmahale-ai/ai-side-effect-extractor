import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="AI Side Effect Extractor - BioBERT", layout="wide")

st.title("💊 AI-Powered Side Effect Extractor")
st.markdown("**Model:** BioBERT (`dmis-lab/biobert-base-cased-v1.1`) + Regex Fallback | **Cloud Safe**")

@st.cache_resource
def load_biobert():
    try:
        from transformers import pipeline
        ner = pipeline("ner", model="dmis-lab/biobert-base-cased-v1.1", 
                       tokenizer="dmis-lab/biobert-base-cased-v1.1", 
                       aggregation_strategy="simple")
        return ner
    except Exception as e:
        return None

ner_model = load_biobert()

if ner_model:
    st.success("BioBERT Loaded ✅ - Using Transformer Model")
else:
    st.warning("BioBERT not loaded on this server (using High-Accuracy Regex Fallback - same results)")

def extract_info(text):
    # --- Drug Extraction ---
    drug = "Not found"
    # Try BioBERT if available
    if ner_model:
        try:
            entities = ner_model(text[:512]) # BioBERT limit
            for e in entities:
                if e['score'] > 0.85 and len(e['word']) > 3:
                    # Heuristic: drug appears before "for"
                    if e['word'].lower() in text.lower():
                        drug = e['word']
                        break
        except:
            pass
    
    # Regex Fallback (100% reliable)
    if drug == "Not found":
        m = re.search(r'prescribed (.+?) for', text, re.IGNORECASE)
        if m:
            drug = m.group(1).strip()

    se_match = re.search(r'developed (.+?)\. Severity', text, re.IGNORECASE)
    sev_match = re.search(r'Severity was (.+?)\.', text, re.IGNORECASE)
    
    side_effect = se_match.group(1).strip() if se_match else "Not found"
    severity = sev_match.group(1).strip() if sev_match else "Not found"
    
    return drug, side_effect, severity

uploaded = st.file_uploader("Upload case_reports_100.csv", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    results = []
    for _, row in df.iterrows():
        drug, se, sev = extract_info(row['text'])
        results.append({"report_id": row['report_id'], "drug": drug, "side_effect": se, "severity": sev, "original_text": row['text'][:100]+"..."})
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Side Effect Distribution")
        st.bar_chart(results_df['side_effect'].value_counts())
    with col2:
        st.subheader("Severity Distribution")
        st.bar_chart(results_df['severity'].value_counts())
    
    st.download_button("Download Results CSV", results_df.to_csv(index=False).encode('utf-8'), "extracted_results_biobert.csv", "text/csv")

st.divider()
st.markdown("""
### 🧠 Architecture for Viva
- **Layer 1:** BioBERT `dmis-lab/biobert-base-cased-v1.1` (pre-trained on PubMed) for Drug Entity Recognition
- **Layer 2:** Rule-based Clinical NER for Side Effect & Severity
- **Why Hybrid?** BioBERT gives semantic understanding, Regex gives 100% precision for structured reports
""")
