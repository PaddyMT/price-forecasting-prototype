import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("🧵 Cotton Price Forecasting Prototype")

uploaded_file = st.file_uploader("📁 Upload Cotton Price Data CSV", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("✅ Data Preview:", data.head())

    varieties = data['Cotton_Variety'].unique().tolist()
    states = data['State'].unique().tolist()

    variety = st.selectbox("Select Cotton Variety", varieties)
    state = st.selectbox("Select State", states)

    df_filtered = data[(data['Cotton_Variety'] == variety) & (data['State'] == state)]
    df_filtered = df_filtered.rename(columns={'Date': 'ds', 'Price_INR_per_Qtl': 'y'})
    df_filtered['ds'] = pd.to_datetime(df_filtered['ds'])

    st.line_chart(df_filtered.set_index('ds')['y'])

    if st.button("📈 Run Forecast"):
        model = Prophet()
        model.fit(df_filtered)
        future = model.make_future_dataframe(periods=12, freq='W')
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)
