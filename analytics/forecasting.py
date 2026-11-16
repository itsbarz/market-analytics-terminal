
import numpy as np
import pandas as pd


def simular_cenarios(
    precos: pd.Series,
    horizontes: tuple = (5, 21, 63, 126, 252),
    numero_simulacoes: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula cenários de preços por bootstrap histórico.

    Sorteia retornos diários observados com reposição e
    compõe trajetórias hipotéticas para cada simulação.

    Retorna:
        resumo: métricas por horizonte.
        faixas: quantis da distribuição de preços projetados.
    """

    if not isinstance(precos, pd.Series):
        raise ValueError("Os preços devem ser uma pandas Series.")

    precos = precos.dropna().astype(float).sort_index()

    if len(precos) < 60:
        raise ValueError("São necessários pelo menos 60 preços.")

    if not np.isfinite(precos.to_numpy()).all():
        raise ValueError("Os preços contêm valores inválidos.")

    if (precos <= 0).any():
        raise ValueError("Os preços devem ser positivos.")

    if not horizontes or any(
        not isinstance(h, (int, np.integer)) or h < 1
        for h in horizontes
    ):
        raise ValueError("Horizontes devem ser inteiros positivos.")

    if len(set(horizontes)) != len(horizontes):
        raise ValueError("Não repita horizontes.")

    if numero_simulacoes < 100:
        raise ValueError("Use pelo menos 100 simulações.")

    retornos = precos.pct_change(fill_method=None).dropna().to_numpy()

    if not np.isfinite(retornos).all():
        raise ValueError("Retornos históricos inválidos.")

    horizonte_maximo = max(horizontes)
    preco_atual = float(precos.iloc[-1])

    rng = np.random.default_rng(seed)

    retornos_sorteados = rng.choice(
        retornos,
        size=(numero_simulacoes, horizonte_maximo),
        replace=True,
    )

    trajetorias = preco_atual * np.cumprod(
        1 + retornos_sorteados,
        axis=1,
    )

    resumo = []

    for horizonte in sorted(horizontes):
        precos_finais = trajetorias[:, horizonte - 1]
        retornos_finais = (precos_finais / preco_atual - 1) * 100

        resumo.append({
            "Horizonte (pregões)": horizonte,
            "Preço esperado": float(np.mean(precos_finais)),
            "Preço mediano": float(np.median(precos_finais)),
            "Retorno esperado (%)": float(np.mean(retornos_finais)),
            "Retorno mediano (%)": float(np.median(retornos_finais)),
            "Percentil 5 (%)": float(np.percentile(retornos_finais, 5)),
            "Percentil 95 (%)": float(np.percentile(retornos_finais, 95)),
            "Probabilidade de perda (%)": float(
                np.mean(retornos_finais < 0) * 100
            ),
        })

    resumo_df = pd.DataFrame(resumo).set_index("Horizonte (pregões)")

    distribuicao = np.vstack([
        np.full(numero_simulacoes, preco_atual),
        trajetorias.T,
    ])

    faixas = pd.DataFrame({
        "P5": np.percentile(distribuicao, 5, axis=1),
        "P25": np.percentile(distribuicao, 25, axis=1),
        "Mediana": np.percentile(distribuicao, 50, axis=1),
        "P75": np.percentile(distribuicao, 75, axis=1),
        "P95": np.percentile(distribuicao, 95, axis=1),
    })

    faixas.index.name = "Pregão futuro"

    return resumo_df, faixas
