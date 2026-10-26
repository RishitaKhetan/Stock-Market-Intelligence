"""Streamlit page for the Stock Market Intelligence dashboard.

This file only draws the UI. All the real work lives in the src/ modules.
"""

import streamlit as st

st.set_page_config(page_title="Stock Market Intelligence", layout="wide")

st.title("Stock Market Intelligence")
st.caption("Educational demo, not financial advice.")

ticker = st.text_input("Stock ticker (e.g. AAPL or TCS.NS)", value="AAPL")
analyze = st.button("Analyze", type="primary")

if analyze:
    st.info(f"Analysis for {ticker.strip().upper()} coming in the next iterations.")
