

import yfinance as yf
import pandas as pd

def obter_dados(ativo: str, periodo: str = "1mo") -> pd.DataFrame:
    dados = yf.download(
        ativo,
        period=periodo,
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    if dados.empty:
        raise ValueError(f"Nenhum dado encontrado para {ativo}")

    return dados