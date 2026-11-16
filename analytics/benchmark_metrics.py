
import numpy as np
import pandas as pd


def calcular_tracking_error(
    retornos_estrategia: pd.Series,
    retornos_benchmark: pd.Series,
) -> float:
    """Tracking Error anualizado, em porcentagem."""

    estrategia, benchmark = retornos_estrategia.align(
        retornos_benchmark,
        join="inner",
    )

    dados = pd.concat([estrategia, benchmark], axis=1).dropna()

    if len(dados) < 2:
        raise ValueError("Dados insuficientes para calcular Tracking Error.")

    retornos_ativos = dados.iloc[:, 0] - dados.iloc[:, 1]

    return float(retornos_ativos.std(ddof=1) * np.sqrt(252))


def calcular_information_ratio(
    retornos_estrategia: pd.Series,
    retornos_benchmark: pd.Series,
) -> float:
    """Information Ratio anualizado."""

    estrategia, benchmark = retornos_estrategia.align(
        retornos_benchmark,
        join="inner",
    )

    dados = pd.concat([estrategia, benchmark], axis=1).dropna()

    if len(dados) < 2:
        raise ValueError("Dados insuficientes para calcular Information Ratio.")

    retornos_ativos = dados.iloc[:, 0] - dados.iloc[:, 1]
    tracking_error_diario = retornos_ativos.std(ddof=1)

    if not np.isfinite(tracking_error_diario) or tracking_error_diario == 0:
        return float("nan")

    return float(
        retornos_ativos.mean()
        / tracking_error_diario
        * np.sqrt(252)
    )


def calcular_rolling_sharpe(
    retornos: pd.Series,
    janela: int = 63,
    taxa_livre_risco_anual: float = 0.0,
) -> pd.Series:
    """Sharpe móvel anualizado usando uma janela de pregões."""

    if janela < 2:
        raise ValueError("A janela deve ser de pelo menos 2 pregões.")

    if not np.isfinite(taxa_livre_risco_anual):
        raise ValueError("Taxa livre de risco inválida.")

    if retornos.empty or not np.isfinite(retornos.to_numpy()).all():
        raise ValueError("Retornos vazios ou inválidos.")

    taxa_diaria = (
        (1 + taxa_livre_risco_anual) ** (1 / 252) - 1
    ) * 100

    excesso = retornos - taxa_diaria

    media = excesso.rolling(
        window=janela,
        min_periods=janela,
    ).mean()

    desvio = excesso.rolling(
        window=janela,
        min_periods=janela,
    ).std(ddof=1)

    rolling_sharpe = (
        media / desvio.replace(0, np.nan)
    ) * np.sqrt(252)

    return rolling_sharpe.rename("Rolling Sharpe")


def calcular_underwater(
    patrimonio: pd.Series,
) -> pd.Series:
    """Distância percentual em relação ao pico histórico do patrimônio."""

    if patrimonio.empty:
        raise ValueError("Série de patrimônio vazia.")

    if not np.isfinite(patrimonio.to_numpy()).all():
        raise ValueError("Patrimônio contém valores inválidos.")

    if (patrimonio <= 0).any():
        raise ValueError("O patrimônio deve ser positivo.")

    picos = patrimonio.cummax()

    underwater = (patrimonio / picos - 1) * 100

    return underwater.rename("Underwater")
