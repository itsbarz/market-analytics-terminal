
import numpy as np
import pandas as pd
import pytest

from analytics.backtesting import executar_backtest_walk_forward


@pytest.fixture
def dados_mercado():
    rng = np.random.default_rng(42)

    datas = pd.bdate_range("2022-01-03", periods=150)

    retornos = rng.normal(
        loc=0.0005,
        scale=0.01,
        size=(150, 3),
    )

    precos = pd.DataFrame(
        100 * np.cumprod(1 + retornos, axis=0),
        index=datas,
        columns=["PETR4", "VALE3", "ITUB4"],
    )

    benchmark = pd.Series(
        100 * np.cumprod(
            1 + rng.normal(0.0003, 0.008, 150)
        ),
        index=datas,
        name="BOVA11",
    )

    return precos, benchmark


def test_backtest_retorna_estrategia_e_benchmark(dados_mercado):
    precos, benchmark = dados_mercado

    resultados, pesos = executar_backtest_walk_forward(
        precos,
        benchmark,
        lookback=30,
        frequencia_rebalanceamento=10,
        numero_carteiras=100,
    )

    assert "Estrategia" in resultados.columns
    assert "Benchmark" in resultados.columns
    assert not resultados.empty
    assert not pesos.empty


def test_pesos_somam_100_porcento(dados_mercado):
    precos, benchmark = dados_mercado

    _, pesos = executar_backtest_walk_forward(
        precos,
        benchmark,
        lookback=30,
        frequencia_rebalanceamento=10,
        numero_carteiras=100,
    )

    assert np.allclose(pesos.sum(axis=1), 1.0)


def test_backtest_nao_utiliza_dados_futuros(dados_mercado):
    precos, benchmark = dados_mercado

    _, pesos_original = executar_backtest_walk_forward(
        precos,
        benchmark,
        lookback=30,
        frequencia_rebalanceamento=10,
        numero_carteiras=100,
    )

    precos_alterados = precos.copy()
    benchmark_alterado = benchmark.copy()

    precos_alterados.iloc[100:] *= 2
    benchmark_alterado.iloc[100:] *= 2

    _, pesos_alterados = executar_backtest_walk_forward(
        precos_alterados,
        benchmark_alterado,
        lookback=30,
        frequencia_rebalanceamento=10,
        numero_carteiras=100,
    )

    datas_anteriores = pesos_original.index[
        pesos_original.index < precos.index[100]
    ]

    pd.testing.assert_frame_equal(
        pesos_original.loc[datas_anteriores],
        pesos_alterados.loc[datas_anteriores],
    )


def test_rejeita_janela_invalida(dados_mercado):
    precos, benchmark = dados_mercado

    with pytest.raises(ValueError):
        executar_backtest_walk_forward(
            precos,
            benchmark,
            lookback=0,
        )
