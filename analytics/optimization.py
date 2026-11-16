
import numpy as np
import pandas as pd


def simular_carteiras(
    retornos: pd.DataFrame,
    numero_carteiras: int = 10000,
    taxa_livre_risco_anual: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simula carteiras long-only usando pesos aleatórios.

    Entrada: retornos diários em porcentagem.
    Saída: retorno e volatilidade anualizados em porcentagem.
    """

    if retornos.empty or len(retornos.columns) < 2:
        raise ValueError("São necessários pelo menos dois ativos.")

    if numero_carteiras < 1:
        raise ValueError("O número de carteiras deve ser positivo.")

    if not np.isfinite(retornos.to_numpy()).all():
        raise ValueError("Os retornos contêm valores ausentes ou inválidos.")

    if not np.isfinite(taxa_livre_risco_anual):
        raise ValueError("A taxa livre de risco deve ser finita.")

    numero_ativos = len(retornos.columns)

    medias_anuais = retornos.mean().to_numpy() * 252
    covariancia_anual = retornos.cov().to_numpy() * 252

    rng = np.random.default_rng(seed)

    pesos = rng.dirichlet(
        np.ones(numero_ativos),
        size=numero_carteiras
    )

    retornos_carteiras = pesos @ medias_anuais

    variancias = np.einsum(
        "ij,jk,ik->i",
        pesos,
        covariancia_anual,
        pesos
    )

    volatilidades = np.sqrt(np.maximum(variancias, 0))

    sharpes = np.divide(
        retornos_carteiras - taxa_livre_risco_anual,
        volatilidades,
        out=np.full(numero_carteiras, np.nan),
        where=volatilidades > 0
    )

    resultado = pd.DataFrame({
        "Retorno anual (%)": retornos_carteiras,
        "Volatilidade anual (%)": volatilidades,
        "Sharpe": sharpes,
    })

    for indice, ativo in enumerate(retornos.columns):
        resultado[f"Peso {ativo} (%)"] = pesos[:, indice] * 100

    return resultado


def obter_carteira_max_sharpe(
    simulacoes: pd.DataFrame
) -> pd.Series:
    if simulacoes.empty or "Sharpe" not in simulacoes.columns:
        raise ValueError("Simulações inválidas ou sem Sharpe.")

    validas = simulacoes.dropna(subset=["Sharpe"])

    if validas.empty:
        raise ValueError("Nenhuma carteira possui Sharpe calculável.")

    return validas.loc[validas["Sharpe"].idxmax()]


def obter_carteira_min_volatilidade(
    simulacoes: pd.DataFrame
) -> pd.Series:
    coluna = "Volatilidade anual (%)"

    if simulacoes.empty or coluna not in simulacoes.columns:
        raise ValueError("Simulações inválidas ou sem volatilidade.")

    validas = simulacoes.dropna(subset=[coluna])

    if validas.empty:
        raise ValueError("Nenhuma carteira possui volatilidade calculável.")

    return validas.loc[validas[coluna].idxmin()]
