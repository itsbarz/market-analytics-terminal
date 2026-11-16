
import numpy as np
import pandas as pd

from analytics.risk_metrics import (
    calcular_volatilidade,
    calcular_drawdown,
    calcular_sharpe,
    calcular_sortino,
    calcular_var,
    calcular_cvar,
)


def validar_pesos(precos: pd.DataFrame, pesos: dict) -> pd.Series:
    if precos.empty:
        raise ValueError("A série de preços está vazia.")

    if not pesos:
        raise ValueError("Informe os pesos da carteira.")

    if set(pesos) != set(precos.columns):
        raise ValueError("Os ativos e os pesos devem corresponder.")

    pesos_series = pd.Series(pesos, dtype=float).reindex(precos.columns)

    if not np.isfinite(pesos_series).all():
        raise ValueError("Os pesos devem ser números finitos.")

    if (pesos_series < 0).any():
        raise ValueError("Pesos negativos não são permitidos.")

    if not np.isclose(pesos_series.sum(), 1.0):
        raise ValueError("A soma dos pesos deve ser igual a 1.")

    return pesos_series


def calcular_retornos_carteira(
    precos: pd.DataFrame,
    pesos: dict
) -> pd.Series:
    pesos_series = validar_pesos(precos, pesos)

    retornos_ativos = precos.pct_change(fill_method=None).dropna(how="any") * 100

    if retornos_ativos.empty:
        raise ValueError("Dados insuficientes para calcular retornos.")

    return retornos_ativos.dot(pesos_series).rename("Portfolio")


def calcular_valor_carteira(
    retornos: pd.Series,
    valor_inicial: float = 10000.0
) -> pd.Series:
    if valor_inicial <= 0:
        raise ValueError("O valor inicial deve ser positivo.")

    if retornos.empty:
        raise ValueError("Não existem retornos para calcular a carteira.")

    return (1 + retornos / 100).cumprod() * valor_inicial


def calcular_correlacao(precos: pd.DataFrame) -> pd.DataFrame:
    if precos.empty or len(precos.columns) < 2:
        raise ValueError("São necessários pelo menos dois ativos.")

    retornos = precos.pct_change(fill_method=None).dropna(how="any")

    return retornos.corr()


def calcular_metricas_carteira(
    retornos: pd.Series,
    valor_inicial: float = 10000.0
) -> dict:
    vol_diaria, vol_anual = calcular_volatilidade(retornos)

    patrimonio = calcular_valor_carteira(retornos, valor_inicial)

    return {
        "retorno_acumulado": float((patrimonio.iloc[-1] / valor_inicial - 1) * 100),
        "volatilidade_diaria": vol_diaria,
        "volatilidade_anual": vol_anual,
        "maximum_drawdown": calcular_drawdown(patrimonio),
        "sharpe": calcular_sharpe(retornos),
        "sortino": calcular_sortino(retornos),
        "var_95": calcular_var(retornos, 0.95),
        "cvar_95": calcular_cvar(retornos, 0.95),
    }
