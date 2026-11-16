
import pandas as pd
import pytest

from analytics.risk_metrics import (
    calcular_retornos,
    calcular_volatilidade,
    calcular_drawdown,
    calcular_sharpe,
    calcular_sortino,
    calcular_var,
    calcular_cvar,
)


def test_retornos():
    precos = pd.Series([100, 110, 99])
    retornos = calcular_retornos(precos)

    assert pd.isna(retornos.iloc[0])
    assert retornos.iloc[1] == pytest.approx(10.0)
    assert retornos.iloc[2] == pytest.approx(-10.0)


def test_volatilidade():
    retornos = pd.Series([1.0, -1.0, 2.0, -2.0])
    diaria, anual = calcular_volatilidade(retornos)

    esperado = retornos.std()

    assert diaria == pytest.approx(esperado)
    assert anual == pytest.approx(esperado * (252 ** 0.5))


def test_drawdown():
    precos = pd.Series([100, 120, 90, 110])
    resultado = calcular_drawdown(precos)

    assert resultado == pytest.approx(-25.0)


def test_sharpe():
    retornos = pd.Series([1.0, 2.0, -1.0, 3.0, 1.5])
    resultado = calcular_sharpe(retornos)

    assert isinstance(resultado, float)


def test_sortino():
    retornos = pd.Series([1.0, 2.0, -1.0, 3.0, -0.5])
    resultado = calcular_sortino(retornos)

    assert isinstance(resultado, float)


def test_var():
    retornos = pd.Series([-5.0, -3.0, -2.0, 1.0, 2.0])
    resultado = calcular_var(retornos, 0.80)

    assert resultado == pytest.approx(-3.4)


def test_cvar():
    retornos = pd.Series([-5.0, -3.0, -2.0, 1.0, 2.0])
    resultado = calcular_cvar(retornos, 0.80)

    assert resultado == pytest.approx(-5.0)
