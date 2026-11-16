
from analytics.market_data import obter_dados
from analytics.risk_metrics import (
    calcular_retornos,
    calcular_volatilidade,
    calcular_drawdown
)


def main():
    print("=== MARKET ANALYTICS TERMINAL ===\n")

    ativo = "PETR4.SA"
    periodo = "1mo"

    print(f"Consultando {ativo}...\n")

    try:
        dados = obter_dados(ativo, periodo)
    except Exception as erro:
        print(f"Erro ao consultar dados: {erro}")
        return

    fechamentos = dados["Close"].iloc[:, 0]

    retornos = calcular_retornos(fechamentos)

    vol_diaria, vol_anual = calcular_volatilidade(retornos)
    max_drawdown = calcular_drawdown(fechamentos)

    print("=== RESUMO FINANCEIRO ===\n")
    print(f"Ativo: {ativo}")
    print(f"Pregões analisados: {len(dados)}")
    print(f"Último fechamento: R$ {fechamentos.iloc[-1]:.2f}")
    print(f"Volatilidade diária: {vol_diaria:.2f}%")
    print(f"Volatilidade anualizada: {vol_anual:.2f}%")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")


if __name__ == "__main__":
    main()
