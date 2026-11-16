
import numpy as np
import pandas as pd
import pytest

from analytics.forecasting import simular_cenarios


@pytest.fixture
def historico():
    rng = np.random.default_rng(42)

    retornos = rng.normal(0.0005, 0.015, 250)

    precos = 100 * np.cumprod(1 + retornos)

    return pd.Series(
        precos,
        index=pd.bdate_range("2025-01-01", periods=250),
    )


def test_forecast_retorna_horizontes(historico):
    resumo, faixas = simular_cenarios(
        historico,
        horizontes=(5, 21, 63),
        numero_simulacoes=1000,
    )

    assert list(resumo.index) == [5, 21, 63]
    assert len(faixas) == 64


def test_forecast_probabilidade_valida(historico):
    resumo, _ = simular_cenarios(
        historico,
        horizontes=(21,),
        numero_simulacoes=1000,
    )

    probabilidade = resumo.loc[21, "Probabilidade de perda (%)"]

    assert 0 <= probabilidade <= 100


def test_forecast_reprodutivel(historico):
    resultado1, _ = simular_cenarios(
        historico,
        horizontes=(21,),
        numero_simulacoes=1000,
        seed=42,
    )

    resultado2, _ = simular_cenarios(
        historico,
        horizontes=(21,),
        numero_simulacoes=1000,
        seed=42,
    )

    pd.testing.assert_frame_equal(resultado1, resultado2)


def test_rejeita_historico_insuficiente():
    precos = pd.Series([100, 101, 102])

    with pytest.raises(ValueError):
        simular_cenarios(precos)


def test_rejeita_horizonte_invalido(historico):
    with pytest.raises(ValueError):
        simular_cenarios(historico, horizontes=(0, 21))


def test_rejeita_simulacoes_insuficientes(historico):
    with pytest.raises(ValueError):
        simular_cenarios(
            historico,
            numero_simulacoes=50,
        )
