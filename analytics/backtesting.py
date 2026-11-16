
import numpy as np
import pandas as pd

from analytics.optimization import (
    simular_carteiras,
    obter_carteira_max_sharpe,
)


def executar_backtest_walk_forward(
    precos: pd.DataFrame,
    benchmark: pd.Series,
    lookback: int = 252,
    frequencia_rebalanceamento: int = 21,
    numero_carteiras: int = 3000,
    taxa_livre_risco_anual: float = 0.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtesting walk-forward com otimização de carteira.

    Estima os pesos usando somente retornos passados e aplica
    esses pesos ao período seguinte.

    Retornos são expressos em porcentagem.
    """

    if precos.empty or len(precos.columns) < 2:
        raise ValueError("Informe pelo menos dois ativos.")

    if precos.columns.has_duplicates:
        raise ValueError("Existem ativos duplicados.")

    if lookback < 2 or lookback >= len(precos):
        raise ValueError("Janela histórica inválida.")

    if frequencia_rebalanceamento < 1:
        raise ValueError("Frequência de rebalanceamento inválida.")

    if numero_carteiras < 1:
        raise ValueError("Número de carteiras inválido.")

    if not isinstance(benchmark, pd.Series):
        raise ValueError("O benchmark deve ser uma Series.")

    precos = precos.sort_index().astype(float)
    benchmark = benchmark.sort_index().astype(float)

    if not np.isfinite(precos.to_numpy()).all():
        raise ValueError("Os preços dos ativos contêm valores inválidos.")

    if not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError("O benchmark contém valores inválidos.")

    if (precos <= 0).any().any() or (benchmark <= 0).any():
        raise ValueError("Os preços devem ser positivos.")

    dados = precos.join(
        benchmark.rename("Benchmark"),
        how="inner",
    ).dropna()

    if len(dados) <= lookback + 1:
        raise ValueError("Histórico insuficiente para o backtest.")

    retornos = dados.pct_change(fill_method=None).dropna() * 100

    retornos_ativos = retornos[precos.columns]
    retornos_benchmark = retornos["Benchmark"]

    resultados = []
    historico_pesos = []

    for inicio in range(
        lookback,
        len(retornos),
        frequencia_rebalanceamento,
    ):
        fim = min(
            inicio + frequencia_rebalanceamento,
            len(retornos),
        )

        # Somente dados anteriores ao período testado.
        janela_treino = retornos_ativos.iloc[
            inicio - lookback:inicio
        ]

        simulacoes = simular_carteiras(
            janela_treino,
            numero_carteiras=numero_carteiras,
            taxa_livre_risco_anual=taxa_livre_risco_anual,
            seed=seed + inicio,
        )

        melhor = obter_carteira_max_sharpe(simulacoes)

        pesos = pd.Series(
            {
                ativo: melhor[f"Peso {ativo} (%)"] / 100
                for ativo in precos.columns
            }
        )

        # Aplicação dos pesos somente no período futuro.
        periodo_teste = retornos_ativos.iloc[inicio:fim]

        retorno_estrategia = periodo_teste.dot(pesos)

        bloco = pd.DataFrame({
            "Estrategia": retorno_estrategia,
            "Benchmark": retornos_benchmark.iloc[inicio:fim],
        })

        resultados.append(bloco)

        historico_pesos.append(
            pd.Series(
                pesos.values,
                index=pesos.index,
                name=periodo_teste.index[0],
            )
        )

    if not resultados:
        raise ValueError("Não foi possível executar o backtest.")

    retornos_backtest = pd.concat(resultados).sort_index()

    pesos_rebalanceamento = pd.DataFrame(historico_pesos)
    pesos_rebalanceamento.index.name = "Data"

    return retornos_backtest, pesos_rebalanceamento
