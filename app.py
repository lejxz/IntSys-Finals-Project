from __future__ import annotations

import streamlit as st

from model.classifier import predict_email


st.set_page_config(page_title="Email Security Scanner", layout="centered")

st.title("Email Security Scanner")
st.write("Detect fraud/phishing and prompt-injection risks using a Naive Bayes classifier.")

text = st.text_area("Email content", height=320, placeholder="Paste the email body here...")

if st.button("Scan"):
    if not text or not text.strip():
        st.error("Please enter email content before scanning.")
    else:
        try:
            result = predict_email(text)
        except FileNotFoundError as exc:
            st.error(f"Dataset error: {exc}")
        except Exception as exc:  # pragma: no cover - surface unexpected errors to user
            st.error(f"Unexpected error: {exc}")
        else:
            # Show key metrics
            safe_pct = int((result.get("is_safe", 0) or 0) * 100)
            fraud_pct = int((result.get("is_fraud", 0) or 0) * 100)
            injection_pct = int((result.get("is_injection", 0) or 0) * 100)

            col1, col2, col3 = st.columns(3)
            col1.metric("Safe", f"{safe_pct}%")
            col2.metric("Fraud", f"{fraud_pct}%")
            col3.metric("Injection", f"{injection_pct}%")

            st.write("**Predicted label:**", result.get("predicted_label"))
            st.write("**Status:**", result.get("status"))

            with st.expander("Raw result"):
                st.json(result)
