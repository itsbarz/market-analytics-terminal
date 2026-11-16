

import pandas as pd

def calcular_retornos(precos: pd.Series) -> pd.Series:
    return precos.pct_change() * 100


def calcular_volatilidade(retornos: pd.Series) -> tuple[float, float]:
    volatilidade_diaria = retornos.std()
    volatitilidade_anual = volatilidade_diaria * (252 ** 0.5)

    return float(volatilidade_diaria), float(volatitilidade_anual)


def calcular_drawdown(precos: pd.Series) -> float:
    pico_acumulado = precos.cummax()
    drawdown = (precos / pico_acumulado - 1) * 100

    return float(drawdown.min())    