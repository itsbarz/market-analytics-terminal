
import streamlit as st
import plotly.graph_objects as go

from analytics.market_data import obter_dados
from analytics.risk_metrics import (
    calcular_retornos,
    calcular_volatilidade,
    calcular_drawdown,
)

st.set_page_config(
    page_title="Market Analytics Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("Market Analytics Terminal")
st.caption("Financial Market Intelligence | Quantitative Analytics")

ativo = st.text_input("Ticker do ativo", value="PETR4.SA")

periodo = st.selectbox(
    "Período histórico",
    ["1mo", "3mo", "6mo", "1y"],
)

@st.cache_data(ttl=300)
def carregar_dados(ticker, periodo):
    return obter_dados(ticker, periodo)

if st.button("Analisar ativo", type="primary"):
    try:
        dados = carregar_dados(ativo, periodo)
        fechamentos = dados["Close"].iloc[:, 0]
        retornos = calcular_retornos(fechamentos)

        vol_diaria, vol_anual = calcular_volatilidade(retornos)
        drawdown = calcular_drawdown(fechamentos)

        st.subheader(f"Análise de {ativo}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Último fechamento",
            f"R$ {fechamentos.iloc[-1]:.2f}"
        )
        col2.metric("Volatilidade diária", f"{vol_diaria:.2f}%")
        col3.metric("Volatilidade anual", f"{vol_anual:.2f}%")
        col4.metric("Maximum Drawdown", f"{drawdown:.2f}%")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fechamentos.index,
            y=fechamentos.values,
            mode="lines",
            name=ativo,
            line=dict(color="#A855F7", width=2),
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Histórico de preços ajustados",
            xaxis_title="Data",
            yaxis_title="Preço (R$)",
            height=480,
        )

        st.plotly_chart(fig, width="stretch")

        st.subheader("Retornos diários (%)")
        st.dataframe(retornos.dropna().round(2))

    except Exception as erro:
        st.error(f"Não foi possível analisar o ativo: {erro}")
