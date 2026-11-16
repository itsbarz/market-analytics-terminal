
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from analytics.market_data import obter_dados
from analytics.risk_metrics import (
    calcular_retornos,
    calcular_volatilidade,
    calcular_drawdown,
    calcular_sharpe,
    calcular_sortino,
    calcular_var,
    calcular_cvar,
    
)

from analytics.optimization import (
    simular_carteiras,
    obter_carteira_max_sharpe,
    obter_carteira_min_volatilidade,

)

from analytics.portfolio import (
    calcular_retornos_carteira,
    calcular_valor_carteira,
    calcular_correlacao,
    calcular_metricas_carteira,
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
    ["1mo", "3mo", "6mo", "1y", "2y", "5y",],
    index=3,
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
        st.caption(f"Período: {periodo} | Pregões analisados: {len(fechamentos)}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Último fechamento",
            f"R$ {fechamentos.iloc[-1]:.2f}"
        )
        col2.metric("Volatilidade diária", f"{vol_diaria:.2f}%")
        col3.metric("Volatilidade anual", f"{vol_anual:.2f}%")
        col4.metric("Maximum Drawdown", f"{drawdown:.2f}%")

        sharpe = calcular_sharpe(retornos)
        sortino = calcular_sortino(retornos)
        var_95 = calcular_var(retornos, 0.95)
        cvar_95 = calcular_cvar(retornos, 0.95)

        st.divider()
        st.subheader("Indicadores de Performance e Risco")

        col5, col6, col7, col8 = st.columns(4)

        col5.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col6.metric("Sortino Ratio", f"{sortino:.2f}")
        col7.metric("Var 95%", f"{var_95:.2f}%")
        col8.metric("CVaR 95%", f"{cvar_95:.2f}%")

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

st.divider()

st.title("Portfolio Analytics")
st.caption("Multi-Asset Portfolio Risk & Performance")

tickers_input = st.text_input(
    "Ativos da carteira",
    value="PETR4.SA, VALE3.SA, ITUB4.SA, BOVA11.SA"
)

pesos_input = st.text_input(
    "Pesos (%) na mesma ordem dos ativos",
    value="25, 25, 25, 25"
)

valor_inicial = st.number_input(
    "Capital inicial (R$)",
    min_value=100.0,
    value=10000.0,
    step=1000.0
)

if st.button("Analisar carteira", type="primary"):
    try:
        tickers = [
            ticker.strip().upper()
            for ticker in tickers_input.split(",")
            if ticker.strip()
        ]

        pesos_lista = [
            float(peso.strip()) / 100
            for peso in pesos_input.split(",")
        ]

        if len(tickers) != len(pesos_lista):
            raise ValueError("Quantidade de pesos diferente da quantidade de ativos.")

        pesos = dict(zip(tickers, pesos_lista))

        precos = {}

        for ticker in tickers:
            dados = carregar_dados(ticker, periodo)
            fechamento = dados["Close"]

            if isinstance(fechamento, pd.DataFrame):
                fechamento = fechamento.iloc[:, 0]

            precos[ticker] = fechamento

        precos_df = pd.DataFrame(precos).dropna()

        retornos_carteira = calcular_retornos_carteira(
            precos_df,
            pesos
        )

        patrimonio = calcular_valor_carteira(
            retornos_carteira,
            valor_inicial
        )

        metricas = calcular_metricas_carteira(
            retornos_carteira,
            valor_inicial
        )

        st.subheader("Performance da carteira")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Patrimônio final",
            f"R$ {patrimonio.iloc[-1]:,.2f}"
        )

        c2.metric(
            "Retorno acumulado",
            f"{metricas['retorno_acumulado']:.2f}%"
        )

        c3.metric(
            "Volatilidade anual",
            f"{metricas['volatilidade_anual']:.2f}%"
        )

        c4.metric(
            "Maximum Drawdown",
            f"{metricas['maximum_drawdown']:.2f}%"
        )

        c5, c6, c7, c8 = st.columns(4)

        c5.metric("Sharpe", f"{metricas['sharpe']:.2f}")
        c6.metric("Sortino", f"{metricas['sortino']:.2f}")
        c7.metric("VaR 95%", f"{metricas['var_95']:.2f}%")
        c8.metric("CVaR 95%", f"{metricas['cvar_95']:.2f}%")

        fig_portfolio = go.Figure()

        fig_portfolio.add_trace(go.Scatter(
            x=patrimonio.index,
            y=patrimonio.values,
            mode="lines",
            name="Portfolio",
            line=dict(color="#A855F7", width=2)
        ))

        fig_portfolio.update_layout(
            template="plotly_dark",
            title="Evolução patrimonial",
            yaxis_title="Patrimônio (R$)",
            height=450
        )

        st.plotly_chart(fig_portfolio, width="stretch")

        st.subheader("Matriz de correlação")

        correlacao = calcular_correlacao(precos_df)

        fig_corr = go.Figure(data=go.Heatmap(
            z=correlacao.values,
            x=correlacao.columns,
            y=correlacao.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=correlacao.round(2).values,
            texttemplate="%{text}"
        ))

        fig_corr.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(fig_corr, width="stretch")

        st.subheader("Alocação da carteira")

        st.dataframe(
            pd.DataFrame({
                "Ativo": tickers,
                "Peso (%)": [peso * 100 for peso in pesos_lista]
            }),
            hide_index=True
        )

    except Exception as erro:
        st.error(f"Erro na análise da carteira: {erro}")

# ==========================================
# PORTFOLIO OPTIMIZER
# ==========================================

st.divider()

st.title("Portfolio Optimizer")
st.caption("Monte Carlo Simulation | Efficient Frontier")

numero_simulacoes = st.number_input(
    "Número de carteiras simuladas",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000,
)

if st.button("Executar otimização"):

    try:
        tickers = list(dict.fromkeys(
            ticker.strip().upper()
            for ticker in tickers_input.split(",")
            if ticker.strip()
        ))

        if len(tickers) < 2:
            st.error("Informe pelo menos dois ativos.")
            st.stop()

        precos = {}

        with st.spinner("Carregando dados e simulando carteiras..."):

            for ticker in tickers:
                dados = carregar_dados(ticker, periodo)
                fechamento = dados["Close"]

                if isinstance(fechamento, pd.DataFrame):
                    fechamento = fechamento.iloc[:, 0]

                precos[ticker] = fechamento

            precos_df = pd.DataFrame(precos).dropna()

            retornos_ativos = (
                precos_df
                .pct_change(fill_method=None)
                .dropna(how="any") * 100
            )

            simulacoes = simular_carteiras(
                retornos_ativos,
                numero_carteiras=int(numero_simulacoes),
            )

            max_sharpe = obter_carteira_max_sharpe(simulacoes)
            min_volatilidade = obter_carteira_min_volatilidade(simulacoes)

        st.success("Otimização concluída!")

        # Fronteira eficiente aproximada
        ordenadas = simulacoes.sort_values("Volatilidade anual (%)")
        retornos_anteriores = (
            ordenadas["Retorno anual (%)"].cummax().shift(1)
        )

        fronteira = ordenadas[
            retornos_anteriores.isna()
            | (ordenadas["Retorno anual (%)"] > retornos_anteriores)
        ]

        # Gráfico
        fig = go.Figure()

        fig.add_trace(go.Scattergl(
            x=simulacoes["Volatilidade anual (%)"],
            y=simulacoes["Retorno anual (%)"],
            mode="markers",
            name="Carteiras simuladas",
            marker=dict(
                size=5,
                color=simulacoes["Sharpe"],
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.55,
            ),
        ))

        fig.add_trace(go.Scatter(
            x=fronteira["Volatilidade anual (%)"],
            y=fronteira["Retorno anual (%)"],
            mode="lines",
            name="Fronteira eficiente aproximada",
            line=dict(color="#A855F7", width=3),
        ))

        fig.add_trace(go.Scatter(
            x=[max_sharpe["Volatilidade anual (%)"]],
            y=[max_sharpe["Retorno anual (%)"]],
            mode="markers",
            name="Melhor Sharpe",
            marker=dict(color="#00E5FF", size=14, symbol="star"),
        ))

        fig.add_trace(go.Scatter(
            x=[min_volatilidade["Volatilidade anual (%)"]],
            y=[min_volatilidade["Retorno anual (%)"]],
            mode="markers",
            name="Mínima volatilidade",
            marker=dict(color="#FFFFFF", size=12, symbol="diamond"),
        ))

        fig.update_layout(
            template="plotly_dark",
            title="Efficient Frontier — Monte Carlo Simulation",
            xaxis_title="Volatilidade anual (%)",
            yaxis_title="Retorno anual esperado (%)",
            height=650,
            legend=dict(orientation="h", y=-0.2),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Resultados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Melhor Sharpe Ratio")

            st.metric(
                "Retorno anual esperado",
                f'{max_sharpe["Retorno anual (%)"]:.2f}%'
            )

            st.metric(
                "Volatilidade anual",
                f'{max_sharpe["Volatilidade anual (%)"]:.2f}%'
            )

            st.metric(
                "Sharpe Ratio",
                f'{max_sharpe["Sharpe"]:.3f}'
            )

        with col2:
            st.subheader("Mínima Volatilidade")

            st.metric(
                "Retorno anual esperado",
                f'{min_volatilidade["Retorno anual (%)"]:.2f}%'
            )

            st.metric(
                "Volatilidade anual",
                f'{min_volatilidade["Volatilidade anual (%)"]:.2f}%'
            )

            st.metric(
                "Sharpe Ratio",
                f'{min_volatilidade["Sharpe"]:.3f}'
            )

        # Pesos recomendados
        st.subheader("Alocação — Melhor Sharpe")

        pesos_otimizados = pd.DataFrame({
            "Ativo": tickers,
            "Peso (%)": [
                max_sharpe[f"Peso {ticker} (%)"]
                for ticker in tickers
            ],
        })

        st.dataframe(
            pesos_otimizados.style.format({"Peso (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as erro:
        st.error(f"Erro durante a otimização: {erro}")