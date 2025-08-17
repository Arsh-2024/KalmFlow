import streamlit as st
import pandas as pd

st.title("Kalman Filter Backtest Results")
RESULTS_PATH = "runs/results.csv"

try:
    df = pd.read_csv(RESULTS_PATH)
    st.dataframe(df)

    st.write("### MAE by Ticker")
    st.bar_chart(df.groupby("ticker")["mae"].mean())
except FileNotFoundError:
    st.error("No results found at 'runs/results.csv'. Run the model first.")
