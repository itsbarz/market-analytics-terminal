
import numpy as np
import pandas as pd
import pytest

from analytics.market_scanner import (
    calcular_indicadores_tecnicos,
    analisar_ativo,
    comparar_ativos,
)


@pytest.fixture
def precos_teste():
    datas = pd.bdate_range("2024-01-01", periods=260)

    rng = np.random.default_rng(42)

    retornos = rng.normal(0.0005, 0.015, 260)

    precos = 100 * np.cumprod(1 + retornos)

    return pd.Series(precos, index=datas, name="TEST")


def test_indicadores_tecnicos(precos_teste):
    resultado = calcular_indicadores_tecnicos(precos_teste)

    colunas = [
        "SMA_20",
        "SMA_50",
        "SMA_200",
        "EMA_12",
        "EMA_26",
        "RSI_14",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
    ]

    for coluna in colunas:
        assert coluna in resultado.columns

    assert resultado["RSI_14"].dropna().between(0, 100).all()


def test_analise_ativo(precos_teste):
    resultado = analisar_ativo(precos_teste)

    assert "Retorno acumulado (%)" in resultado
    assert "Volatilidade anual (%)" in resultado
    assert "Sharpe" in resultado
    assert "Maximum Drawdown (%)" in resultado


def test_beta_com_benchmark(precos_teste):
    benchmark = precos_teste * 0.95

    resultado = analisar_ativo(
        precos_teste,
        benchmark,
    )

    assert "Beta" in resultado
    assert resultado["Beta"] == pytest.approx(1.0)


def test_comparar_ativos(precos_teste):
    dados = pd.DataFrame({
        "ATIVO_A": precos_teste,
        "ATIVO_B": precos_teste * 1.2,
    })

    resultado = comparar_ativos(dados)

    assert len(resultado) == 2
    assert "Sharpe" in resultado.columns


def test_rejeita_historico_insuficiente():
    precos = pd.Series([100.0])

    with pytest.raises(ValueError):
        analisar_ativo(precos)


def test_rejeita_precos_invalidos():
    precos = pd.Series([100.0, -10.0, 105.0])

    with pytest.raises(ValueError):
        analisar_ativo(precos)


def test_rejeita_comparacao_com_um_ativo(precos_teste):
    dados = pd.DataFrame({"ATIVO_A": precos_teste})

    with pytest.raises(ValueError):
        comparar_ativos(dados)
