
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

from analytics.backtesting import executar_backtest_walk_forward

from analytics.risk_metrics import (
    calcular_volatilidade,
    calcular_drawdown,
    calcular_sharpe,
    calcular_sortino,
)

from analytics.benchmark_metrics import (
    calcular_tracking_error,
    calcular_information_ratio,
    calcular_rolling_sharpe,
    calcular_underwater,
)

from analytics.forecasting import simular_cenarios

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


# ==========================================
# BACKTESTING LAB
# ==========================================

st.divider()

st.title("Backtesting Lab")
st.caption("Walk-Forward Analysis | Strategy vs Benchmark")

tickers_backtest = st.text_input(
    "Ativos da estratégia",
    value="PETR4.SA, VALE3.SA, ITUB4.SA",
    key="backtest_tickers",
)

benchmark_ticker = st.text_input(
    "Benchmark",
    value="BOVA11.SA",
    key="backtest_benchmark",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    lookback = st.number_input(
        "Janela de treinamento (pregões)",
        min_value=60,
        max_value=1000,
        value=252,
    )

with col2:
    rebalanceamento = st.number_input(
        "Rebalanceamento (pregões)",
        min_value=5,
        max_value=252,
        value=21,
    )

with col3:
    simulacoes_backtest = st.number_input(
        "Carteiras por janela",
        min_value=100,
        max_value=20000,
        value=3000,
        step=100,
    )

with col4:
    custo_total_bps = st.number_input(
        "Custo total (bps)",
        min_value=0.0,
        max_value=200.0,
        value=10.0,
        step=1.0,
    )

if st.button("Executar Backtest", key="run_backtest"):

    try:
        tickers = list(dict.fromkeys(
            ticker.strip().upper()
            for ticker in tickers_backtest.split(",")
            if ticker.strip()
        ))

        benchmark_nome = benchmark_ticker.strip().upper()

        if len(tickers) < 2:
            st.error("Informe pelo menos dois ativos.")
            st.stop()

        if benchmark_nome in tickers:
            st.error("O benchmark não deve estar entre os ativos da estratégia.")
            st.stop()

        with st.spinner("Executando análise walk-forward..."):

            precos = {}

            for ticker in tickers:
                dados = carregar_dados(ticker, "5y")
                fechamento = dados["Close"]

                if isinstance(fechamento, pd.DataFrame):
                    fechamento = fechamento.iloc[:, 0]

                precos[ticker] = fechamento

            dados_benchmark = carregar_dados(benchmark_nome, "5y")
            benchmark = dados_benchmark["Close"]

            if isinstance(benchmark, pd.DataFrame):
                benchmark = benchmark.iloc[:, 0]

            precos_df = pd.DataFrame(precos).dropna()

            resultados, historico_pesos = executar_backtest_walk_forward(
                precos=precos_df,
                benchmark=benchmark,
                lookback=int(lookback),
                frequencia_rebalanceamento=int(rebalanceamento),
                numero_carteiras=int(simulacoes_backtest),
                custo_total_bps=float(custo_total_bps),
            )

            patrimonio = (
    10000
    * (1 + resultados[["Estrategia", "Benchmark"]] / 100).cumprod()
)

            patrimonio.columns = ["Estratégia", "Benchmark"]

            retorno_estrategia = resultados["Estrategia"]
            retorno_benchmark = resultados["Benchmark"]

            retorno_acumulado = (
                patrimonio["Estratégia"].iloc[-1] / 10000 - 1
            ) * 100

            retorno_benchmark_acumulado = (
                patrimonio["Benchmark"].iloc[-1] / 10000 - 1
            ) * 100

            turnover_medio = resultados["Turnover"].mean()
            custos_acumulados = resultados["Custos"].sum()

            vol_diaria, vol_anual = calcular_volatilidade(
                retorno_estrategia
            )

            sharpe = calcular_sharpe(retorno_estrategia)
            sortino = calcular_sortino(retorno_estrategia)

            drawdown = calcular_drawdown(
                patrimonio["Estratégia"]
            )

            drawdown_benchmark = calcular_drawdown(
                patrimonio["Benchmark"]
            )

            drawdown_serie = (
                patrimonio["Estratégia"]
                / patrimonio["Estratégia"].cummax() - 1
            ) * 100

            tracking_error = calcular_tracking_error(
                retorno_estrategia,
                retorno_benchmark,
            )

            information_ratio = calcular_information_ratio(
                retorno_estrategia,
                retorno_benchmark,
            )

            rolling_sharpe = calcular_rolling_sharpe(
                retorno_estrategia,
                janela=63,
            )

            underwater = calcular_underwater(
                patrimonio["Estratégia"]
            )


        st.success("Backtest concluído!")

        # PERFORMANCE
        st.subheader("Performance Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Retorno da estratégia",
            f"{retorno_acumulado:.2f}%"
        )

        c2.metric(
            "Retorno do benchmark",
            f"{retorno_benchmark_acumulado:.2f}%"
        )

        c3.metric(
            "Sharpe Ratio",
            f"{sharpe:.3f}"
        )

        c4.metric(
            "Maximum Drawdown",
            f"{drawdown:.2f}%"
        )

        # EQUITY CURVE
        st.subheader("Equity Curve")

        fig_equity = go.Figure()

        fig_equity.add_trace(go.Scatter(
            x=patrimonio.index,
            y=patrimonio["Estratégia"],
            name="Estratégia",
            line=dict(color="#A855F7", width=3),
        ))

        fig_equity.add_trace(go.Scatter(
            x=patrimonio.index,
            y=patrimonio["Benchmark"],
            name="Benchmark",
            line=dict(color="#00E5FF", width=2),
        ))

        fig_equity.update_layout(
            template="plotly_dark",
            height=450,
            yaxis_title="Patrimônio (R$)",
            xaxis_title="Data",
            hovermode="x unified",
        )

        st.plotly_chart(fig_equity, use_container_width=True)

        # DRAWDOWN
        st.subheader("Drawdown Analysis")

        fig_drawdown = go.Figure()

        fig_drawdown.add_trace(go.Scatter(
            x=drawdown_serie.index,
            y=drawdown_serie,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#EF4444"),
        ))

        fig_drawdown.update_layout(
            template="plotly_dark",
            height=300,
            yaxis_title="Drawdown (%)",
        )

        st.plotly_chart(fig_drawdown, use_container_width=True)

        # RISK METRICS
        st.subheader("Risk Analytics")

        r1, r2, r3, r4, = st.columns(4)


        r1.metric("Volatilidade anual", f"{vol_anual:.2f}%")
        r2.metric("Sharpe", f"{sharpe:.3f}")
        r3.metric("Sortino", f"{sortino:.3f}")
        r4.metric(
            "Benchmark Max Drawdown",
            f"{drawdown_benchmark:.2f}%"
        )

        r5, r6, = st.columns(2)

        r5.metric(
            "Turnover médio por pregão",
            f"{turnover_medio:.3f}%"
        )

        r6.metric(
            "Custos acumulados",
            f"{custos_acumulados:.3f}%"
        )

        # ALLOCATION HISTORY
        st.subheader("Portfolio Allocation History")

        pesos_percentuais = historico_pesos * 100

        fig_pesos = go.Figure()

        for ativo in pesos_percentuais.columns:
            fig_pesos.add_trace(go.Scatter(
                x=pesos_percentuais.index,
                y=pesos_percentuais[ativo],
                name=ativo,
                stackgroup="allocation",
                mode="lines",
            ))

        fig_pesos.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Alocação (%)",
            hovermode="x unified",
        )

        st.plotly_chart(fig_pesos, use_container_width=True)

        st.subheader("Rebalanceamento History")

        st.dataframe(
            pesos_percentuais.style.format("{:.2f}%"),
            use_container_width=True,
        )

        st.caption(
            "Modelo experimental: não inclui custos, impostos, "
            "slippage ou restrições de concentração. "
            "Resultados históricos não garantem retornos futuros."
        )

    except Exception as erro:
        st.error(f"Erro no backtesting: {erro}")


# ==========================================
# FORECASTING LAB
# ==========================================

st.divider()

st.title("Forecasting Lab")
st.caption(
    "Probabilistic Market Scenarios | Historical Bootstrap"
)

st.markdown(
    """
    <div style="
        background: linear-gradient(110deg, #21152e, #171c2b);
        border: 1px solid #7041a6;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 20px;
    ">
        <div style="color:#C084FC; font-size:12px; letter-spacing:2px;">
            MARKET SCENARIO ENGINE
        </div>
        <div style="font-size:25px; font-weight:700; color:white;">
            Explore possible market futures.
        </div>
        <div style="color:#B8B8C5; font-size:13px;">
            Bootstrap simulation · Return distributions · Risk scenarios
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

fc1, fc2, fc3 = st.columns([2, 2, 1])

with fc1:
    ticker_forecast = st.text_input(
        "Asset / Ticker",
        value="PETR4.SA",
        key="forecast_ticker",
    )

with fc2:
    periodo_forecast = st.selectbox(
        "Historical data",
        ["1y", "2y", "5y", "10y"],
        index=2,
        key="forecast_period",
    )

with fc3:
    numero_forecasts = st.number_input(
        "Simulations",
        min_value=1000,
        max_value=50000,
        value=10000,
        step=1000,
        key="forecast_simulations",
    )

if st.button("Generate Market Scenarios", key="run_forecast"):

    try:
        with st.spinner("Simulating possible market scenarios..."):

            ticker = ticker_forecast.strip().upper()

            if not ticker:
                raise ValueError("Informe um ticker válido.")

            dados_forecast = carregar_dados(
                ticker,
                periodo_forecast,
            )

            fechamento = dados_forecast["Close"]

            if isinstance(fechamento, pd.DataFrame):
                fechamento = fechamento.iloc[:, 0]

            resumo_forecast, faixas_forecast = simular_cenarios(
                fechamento,
                horizontes=(5, 21, 63, 126, 252),
                numero_simulacoes=int(numero_forecasts),
                seed=42,
            )

            preco_atual = float(fechamento.iloc[-1])

        st.success(f"Scenario analysis completed: {ticker}")

        # ==========================================
        # CURRENT MARKET SNAPSHOT
        # ==========================================

        st.subheader("Market Snapshot")

        m1, m2, m3 = st.columns(3)

        m1.metric("Current Price", f"{preco_atual:.2f}")
        m2.metric("Historical Observations", f"{len(fechamento)}")
        m3.metric("Monte Carlo Paths", f"{int(numero_forecasts):,}")

        # ==========================================
        # FORECAST HORIZON SELECTOR
        # ==========================================

        st.subheader("Forecast Horizon")

        horizonte_selecionado = st.select_slider(
            "Projection horizon",
            options=[5, 21, 63, 126, 252],
            value=21,
            format_func=lambda x: {
                5: "1 Week",
                21: "1 Month",
                63: "3 Months",
                126: "6 Months",
                252: "12 Months",
            }[x],
        )

        cenario = resumo_forecast.loc[horizonte_selecionado]

        # ==========================================
        # PROBABILISTIC METRICS
        # ==========================================

        st.subheader("Probability & Return Analysis")

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Expected Return",
            f'{cenario["Retorno esperado (%)"]:.2f}%',
        )

        k2.metric(
            "Median Return",
            f'{cenario["Retorno mediano (%)"]:.2f}%',
        )

        k3.metric(
            "Probability of Loss",
            f'{cenario["Probabilidade de perda (%)"]:.1f}%',
        )

        k4.metric(
            "Median Price",
            f'{cenario["Preço mediano"]:.2f}',
        )

        # ==========================================
        # PRICE DISTRIBUTION
        # ==========================================

        st.subheader("Projected Price Distribution")

        st.caption(
            "Shaded regions represent simulated percentile ranges, "
            "not confidence guarantees."
        )

        fig_forecast = go.Figure()

        x = faixas_forecast.index

        fig_forecast.add_trace(go.Scatter(
            x=x,
            y=faixas_forecast["P95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_forecast.add_trace(go.Scatter(
            x=x,
            y=faixas_forecast["P5"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(168,85,247,0.12)",
            name="5th–95th percentile",
        ))

        fig_forecast.add_trace(go.Scatter(
            x=x,
            y=faixas_forecast["P75"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig_forecast.add_trace(go.Scatter(
            x=x,
            y=faixas_forecast["P25"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(168,85,247,0.24)",
            name="25th–75th percentile",
        ))

        fig_forecast.add_trace(go.Scatter(
            x=x,
            y=faixas_forecast["Mediana"],
            mode="lines",
            name="Median scenario",
            line=dict(color="#C084FC", width=3),
        ))

        fig_forecast.add_hline(
            y=preco_atual,
            line_dash="dash",
            line_color="#67E8F9",
            annotation_text="Current price",
        )

        fig_forecast.update_layout(
            template="plotly_dark",
            height=520,
            xaxis_title="Future trading sessions",
            yaxis_title="Simulated price",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                y=-0.2,
            ),
            margin=dict(l=20, r=20, t=35, b=60),
        )

        st.plotly_chart(
            fig_forecast,
            use_container_width=True,
        )

        # ==========================================
        # ALL HORIZONS TABLE
        # ==========================================

        st.subheader("Forecast Scenarios by Horizon")

        tabela_forecast = resumo_forecast.copy()

        tabela_forecast.index = [
            "1 Week",
            "1 Month",
            "3 Months",
            "6 Months",
            "12 Months",
        ]

        tabela_forecast = resumo_forecast.rename(columns={
            "Preço esperado": "Expected Price",
            "Preço mediano": "Median Price",
            "Retorno esperado (%)": "Expected Return",
            "Retorno mediano (%)": "Median Return",
            "Percentil 5 (%)": "P5 Return",
            "Percentil 95 (%)": "P95 Return",
            "Probabilidade de perda (%)": "Loss Probability",
        })

        st.dataframe(
            tabela_forecast.style.format({
                "Expected Price": "{:.2f}",
                "Median Price": "{:.2f}",
                "Expected Return": "{:.2f}%",
                "Median Return": "{:.2f}%",
                "P5 Return": "{:.2f}%",
                "P95 Return": "{:.2f}%",
                "Loss Probability": "{:.2f}%",
            }),
            use_container_width=True,
        )


        # ==========================================
        # MARKET OUTLOOK
        # ==========================================

        st.divider()
        st.subheader("Market Outlook")
        st.caption("Bear, Base and Bull scenarios")

        cenarios = pd.DataFrame({
            "Bear Case": preco_atual * (
                1 + resumo_forecast["Percentil 5 (%)"] / 100
            ),
            "Base Case": preco_atual * (
                1 + resumo_forecast["Retorno mediano (%)"] / 100
            ),
            "Bull Case": preco_atual * (
                1 + resumo_forecast["Percentil 95 (%)"] / 100
            ),
            "Loss Probability": resumo_forecast[
                "Probabilidade de perda (%)"
            ],
        })

        cenarios.index = [
            "1 Week", "1 Month", "3 Months", "6 Months", "12 Months"
        ]

        st.dataframe(
            cenarios.style.format({
                "Bear Case": "{:.2f}",
                "Base Case": "{:.2f}",
                "Bull Case": "{:.2f}",
                "Loss Probability": "{:.2f}%",
            }),
            use_container_width=True,
        )


        st.caption(
            "Research model: historical bootstrap with replacement. "
            "Assumes historical daily returns are representative of "
            "future scenarios. Does not account for transaction costs, "
            "regime changes, market impact or future information. "
            "Not investment advice."
        )

    except Exception as erro:
        st.error(f"Forecasting error: {erro}")
