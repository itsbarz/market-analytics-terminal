
import pandas as pd
import pytest

from analytics.portfolio import (
    validar_pesos,
    calcular_retornos_carteira,
    calcular_valor_carteira,
    calcular_correlacao,
    calcular_metricas_carteira,
)


@pytest.fixture
def precos():
    return pd.DataFrame({
        "PETR4.SA": [100, 110, 105, 115, 120],
        "VALE3.SA": [50, 52, 51, 54, 56],
        "ITUB4.SA": [30, 31, 32, 31, 33],
    })


@pytest.fixture
def pesos():
    return {
        "PETR4.SA": 0.40,
        "VALE3.SA": 0.35,
        "ITUB4.SA": 0.25,
    }


def test_validar_pesos(precos, pesos):
    resultado = validar_pesos(precos, pesos)
    assert resultado.sum() == pytest.approx(1.0)


def test_pesos_invalidos(precos):
    with pytest.raises(ValueError):
        validar_pesos(precos, {"PETR4.SA": 0.5, "VALE3.SA": 0.5})


def test_retornos_carteira(precos, pesos):
    resultado = calcular_retornos_carteira(precos, pesos)
    assert len(resultado) == 4
    assert resultado.name == "Portfolio"


def test_valor_carteira():
    retornos = pd.Series([10.0, -5.0])
    resultado = calcular_valor_carteira(retornos, 10000)
    assert resultado.iloc[-1] == pytest.approx(10450)


def test_correlacao(precos):
    resultado = calcular_correlacao(precos)
    assert resultado.shape == (3, 3)


def test_metricas_carteira(precos, pesos):
    retornos = calcular_retornos_carteira(precos, pesos)
    resultado = calcular_metricas_carteira(retornos)

    assert "sharpe" in resultado
    assert "maximum_drawdown" in resultado
    assert "var_95" in resultado
