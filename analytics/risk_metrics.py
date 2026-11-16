
import numpy as np
import pandas as pd


def calcular_retornos(precos: pd.Series) -> pd.Series:
    return precos.pct_change() * 100


def calcular_volatilidade(retornos: pd.Series) -> tuple[float, float]:
    volatilidade_diaria = retornos.std()
    volatilidade_anual = volatilidade_diaria * np.sqrt(252)

    return float(volatilidade_diaria), float(volatilidade_anual)


def calcular_drawdown(precos: pd.Series) -> float:
    pico_acumulado = precos.cummax()
    drawdown = (precos / pico_acumulado - 1) * 100

    return float(drawdown.min())


def calcular_sharpe(
    retornos: pd.Series,
    taxa_livre_risco_anual: float = 0.0
) -> float:
    retornos_validos = retornos.dropna()

    taxa_diaria = (1 + taxa_livre_risco_anual) ** (1 / 252) - 1
    taxa_diaria *= 100

    excesso_retorno = retornos_validos - taxa_diaria
    volatilidade = excesso_retorno.std()

    if volatilidade == 0 or pd.isna(volatilidade):
        return float("nan")

    return float(
        (excesso_retorno.mean() / volatilidade) * np.sqrt(252)
    )


def calcular_sortino(
    retornos: pd.Series,
    taxa_livre_risco_anual: float = 0.0
) -> float:
    retornos_validos = retornos.dropna()

    taxa_diaria = ((1 + taxa_livre_risco_anual) ** (1 / 252) - 1) * 100
    excesso_retorno = retornos_validos - taxa_diaria

    retornos_negativos = excesso_retorno[excesso_retorno < 0]

    downside_deviation = np.sqrt((retornos_negativos ** 2).mean())

    if downside_deviation == 0 or pd.isna(downside_deviation):
        return float("nan")

    return float(
        (excesso_retorno.mean() / downside_deviation) * np.sqrt(252)
    )


def calcular_var(
    retornos: pd.Series,
    confianca: float = 0.95
) -> float:
    retornos_validos = retornos.dropna()

    if not 0 < confianca < 1:
        raise ValueError("A confiança deve estar entre 0 e 1.")

    return float(np.percentile(retornos_validos, (1 - confianca) * 100))


def calcular_cvar(
    retornos: pd.Series,
    confianca: float = 0.95
) -> float:
    retornos_validos = retornos.dropna()

    var = calcular_var(retornos_validos, confianca)
    perdas_extremas = retornos_validos[retornos_validos <= var]

    if perdas_extremas.empty:
        return float("nan")

    return float(perdas_extremas.mean())
