
import numpy as np
import pandas as pd
import pytest

from analytics.optimization import (
    simular_carteiras,
    obter_carteira_max_sharpe,
    obter_carteira_min_volatilidade,
)


@pytest.fixture
def retornos():
    return pd.DataFrame({
        "PETR4.SA": [1.0, -0.5, 1.2, -0.8, 0.7, 1.1],
        "VALE3.SA": [0.4, 0.2, -0.3, 0.5, 0.6, -0.1],
        "ITUB4.SA": [0.3, 0.1, 0.2, -0.1, 0.4, 0.2],
    })


def test_simular_carteiras(retornos):
    resultado = simular_carteiras(
        retornos,
        numero_carteiras=1000
    )

    assert len(resultado) == 1000
    assert "Sharpe" in resultado.columns
    assert "Retorno anual (%)" in resultado.columns
    assert "Volatilidade anual (%)" in resultado.columns


def test_pesos_somam_100(retornos):
    resultado = simular_carteiras(retornos, 1000)

    colunas_pesos = [
        coluna for coluna in resultado.columns
        if coluna.startswith("Peso ")
    ]

    somas = resultado[colunas_pesos].sum(axis=1)

    assert np.allclose(somas, 100.0)


def test_carteira_max_sharpe(retornos):
    simulacoes = simular_carteiras(retornos, 1000)
    melhor = obter_carteira_max_sharpe(simulacoes)

    assert melhor["Sharpe"] == pytest.approx(
        simulacoes["Sharpe"].max()
    )


def test_carteira_min_volatilidade(retornos):
    simulacoes = simular_carteiras(retornos, 1000)
    melhor = obter_carteira_min_volatilidade(simulacoes)

    assert melhor["Volatilidade anual (%)"] == pytest.approx(
        simulacoes["Volatilidade anual (%)"].min()
    )


def test_rejeita_menos_de_dois_ativos():
    retornos = pd.DataFrame({"PETR4.SA": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        simular_carteiras(retornos)


def test_rejeita_retornos_invalidos():
    retornos = pd.DataFrame({
        "PETR4.SA": [1.0, np.nan, 3.0],
        "VALE3.SA": [2.0, 1.0, 2.0],
    })

    with pytest.raises(ValueError):
        simular_carteiras(retornos)
