
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
    custo_total_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward backtest com custos de transação.

    custo_total_bps representa o custo combinado estimado
    de execução, taxas e slippage, em basis points.

    100 bps = 1%.
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

    if not np.isfinite(custo_total_bps) or custo_total_bps < 0:
        raise ValueError("O custo em bps deve ser finito e não negativo.")

    if not isinstance(benchmark, pd.Series):
        raise ValueError("O benchmark deve ser uma Series.")

    precos = precos.sort_index().astype(float)
    benchmark = benchmark.sort_index().astype(float)

    if not np.isfinite(precos.to_numpy()).all():
        raise ValueError("Preços dos ativos inválidos.")

    if not np.isfinite(benchmark.to_numpy()).all():
        raise ValueError("Preços do benchmark inválidos.")

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

    pesos_atuais = None

    for inicio in range(
        lookback,
        len(retornos),
        frequencia_rebalanceamento,
    ):
        fim = min(
            inicio + frequencia_rebalanceamento,
            len(retornos),
        )

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

        novos_pesos = pd.Series({
            ativo: melhor[f"Peso {ativo} (%)"] / 100
            for ativo in precos.columns
        })

        if pesos_atuais is None:
            # Primeira alocação: carteira parte de caixa.
            turnover = 1.0
        else:
            turnover = float(
                (novos_pesos - pesos_atuais).abs().sum() / 2
            )

        periodo_teste = retornos_ativos.iloc[inicio:fim]

        retorno_bruto = periodo_teste.dot(novos_pesos)

        custo_pct = turnover * custo_total_bps / 100

        retorno_liquido = retorno_bruto.copy()

        # O custo é aplicado no primeiro pregão após o rebalanceamento.
        retorno_liquido.iloc[0] -= custo_pct

        bloco = pd.DataFrame({
            "Estrategia": retorno_liquido,
            "Benchmark": retornos_benchmark.iloc[inicio:fim],
            "Turnover": 0.0,
            "Custos (%)": 0.0,
        })

        bloco.iloc[0, bloco.columns.get_loc("Turnover")] = turnover * 100
        bloco.iloc[0, bloco.columns.get_loc("Custos (%)")] = custo_pct

        resultados.append(bloco)

        historico_pesos.append(
            pd.Series(
                novos_pesos.values,
                index=novos_pesos.index,
                name=periodo_teste.index[0],
            )
        )

        # Atualiza os pesos pela valorização dos ativos no período.
        crescimento = (1 + periodo_teste / 100).prod()

        pesos_valorizados = novos_pesos * crescimento

        pesos_atuais = pesos_valorizados / pesos_valorizados.sum()

    if not resultados:
        raise ValueError("Não foi possível executar o backtest.")

    retornos_backtest = pd.concat(resultados).sort_index()

    pesos_rebalanceamento = pd.DataFrame(historico_pesos)
    pesos_rebalanceamento.index.name = "Data"

    return retornos_backtest, pesos_rebalanceamento
