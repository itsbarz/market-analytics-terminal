
import numpy as np
import pandas as pd


def _validar_precos(precos: pd.Series) -> pd.Series:
    if not isinstance(precos, pd.Series):
        raise ValueError("Os preços devem ser uma pandas Series.")

    precos = precos.dropna().astype(float).sort_index()

    if len(precos) < 2:
        raise ValueError("Histórico insuficiente.")

    if not np.isfinite(precos.to_numpy()).all():
        raise ValueError("Existem preços inválidos.")

    if (precos <= 0).any():
        raise ValueError("Os preços devem ser positivos.")

    return precos


def calcular_indicadores_tecnicos(
    precos: pd.Series,
) -> pd.DataFrame:
    precos = _validar_precos(precos)

    indicadores = pd.DataFrame(index=precos.index)
    indicadores["Close"] = precos

    # Médias móveis
    indicadores["SMA_20"] = precos.rolling(20).mean()
    indicadores["SMA_50"] = precos.rolling(50).mean()
    indicadores["SMA_200"] = precos.rolling(200).mean()

    indicadores["EMA_12"] = precos.ewm(
        span=12,
        adjust=False,
    ).mean()

    indicadores["EMA_26"] = precos.ewm(
        span=26,
        adjust=False,
    ).mean()

    # RSI 14 — método de suavização de Wilder
    variacao = precos.diff()

    ganhos = variacao.clip(lower=0)
    perdas = -variacao.clip(upper=0)

    media_ganhos = ganhos.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    media_perdas = perdas.ewm(
        alpha=1 / 14,
        adjust=False,
        min_periods=14,
    ).mean()

    rs = media_ganhos / media_perdas.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (media_perdas == 0) & (media_ganhos > 0),
        100,
    )

    rsi = rsi.mask(
        (media_perdas == 0) & (media_ganhos == 0),
        50,
    )

    indicadores["RSI_14"] = rsi

    # MACD
    macd = indicadores["EMA_12"] - indicadores["EMA_26"]

    sinal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()

    indicadores["MACD"] = macd
    indicadores["MACD_Signal"] = sinal
    indicadores["MACD_Histogram"] = macd - sinal

    # Retorno diário em porcentagem
    indicadores["Daily_Return (%)"] = (
        precos.pct_change(fill_method=None) * 100
    )

    return indicadores


def analisar_ativo(
    precos: pd.Series,
    benchmark: pd.Series | None = None,
) -> dict:
    precos = _validar_precos(precos)

    retornos = precos.pct_change(fill_method=None).dropna()

    if retornos.empty:
        raise ValueError("Não existem retornos suficientes.")

    patrimonio = (1 + retornos).cumprod()

    retorno_acumulado = (
        (precos.iloc[-1] / precos.iloc[0]) - 1
    ) * 100

    dias = (precos.index[-1] - precos.index[0]).days

    cagr = (
        ((precos.iloc[-1] / precos.iloc[0]) ** (365.25 / dias) - 1) * 100
        if dias > 0 else np.nan
    )

    volatilidade = retornos.std() * np.sqrt(252) * 100

    sharpe = (
        retornos.mean() / retornos.std() * np.sqrt(252)
        if retornos.std() > 0 else np.nan
    )

    drawdown = (patrimonio / patrimonio.cummax() - 1) * 100

    resultado = {
        "Último preço": float(precos.iloc[-1]),
        "Retorno acumulado (%)": float(retorno_acumulado),
        "CAGR (%)": float(cagr),
        "Volatilidade anual (%)": float(volatilidade),
        "Sharpe": float(sharpe),
        "Maximum Drawdown (%)": float(drawdown.min() * 1.0),
        "Observações": int(len(precos)),
    }

    if benchmark is not None:
        benchmark = _validar_precos(benchmark)

        retornos_benchmark = benchmark.pct_change(
            fill_method=None
        ).rename("benchmark")

        dados = pd.concat(
            [
                retornos.rename("ativo"),
                retornos_benchmark,
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(dados) < 2:
            raise ValueError("Dados insuficientes para calcular o beta.")

        variancia_benchmark = dados["benchmark"].var()

        beta = (
            dados["ativo"].cov(dados["benchmark"])
            / variancia_benchmark
            if variancia_benchmark > 0 else np.nan
        )

        resultado["Beta"] = float(beta)

    return resultado


def comparar_ativos(
    precos: pd.DataFrame,
    benchmark: pd.Series | None = None,
) -> pd.DataFrame:
    if not isinstance(precos, pd.DataFrame):
        raise ValueError("Os preços devem estar em um DataFrame.")

    if precos.empty or len(precos.columns) < 2:
        raise ValueError("Informe pelo menos dois ativos.")

    if precos.columns.has_duplicates:
        raise ValueError("Existem ativos duplicados.")

    resultados = {}

    for ativo in precos.columns:
        resultados[ativo] = analisar_ativo(
            precos[ativo],
            benchmark=benchmark,
        )

    return pd.DataFrame(resultados).T
