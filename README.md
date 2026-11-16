# Market Analytics Terminal

**Quantitative Financial Analytics | Python | Market Risk**

A quantitative financial analysis platform designed to retrieve market data, calculate risk metrics and provide analytical insights through an interactive dashboard.

## Overview

Market Analytics Terminal is a Python-based project focused on financial market analysis and quantitative risk measurement.

The initial version provides historical market data retrieval, financial metrics calculation and interactive data visualization.

## Current Features

- Historical market data retrieval using Yahoo Finance
- Daily percentage returns calculation
- Daily and annualized volatility
- Maximum Drawdown calculation
- Interactive dashboard built with Streamlit
- Financial data visualization with Plotly
- Automated tests with Pytest

## Technology Stack

- Python
- Pandas
- NumPy
- yfinance
- Streamlit
- Plotly
- Pytest

## Project Structure

```text
market-analytics-terminal/
|-- analytics/
|   |-- __init__.py
|   |-- market_data.py
|   |-- risk_metrics.py
|-- tests/
|   |-- test_risk_metrics.py
|-- dashboard.py
|-- main.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/itsbarz/market-analytics-terminal.git
cd market-analytics-terminal
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

```bash
python -m streamlit run dashboard.py
```

## Running Tests

```bash
python -m pytest -v
```

## Development Roadmap

- [x] Historical market data retrieval
- [x] Basic quantitative risk metrics
- [x] Interactive dashboard
- [x] Initial automated tests
- [ ] Portfolio analytics
- [ ] Sharpe Ratio and Sortino Ratio
- [ ] Value at Risk (VaR) and Conditional VaR
- [ ] Correlation analysis
- [ ] Backtesting engine
- [ ] Database integration
- [ ] REST API
- [ ] Advanced portfolio risk management

## Disclaimer

This project is intended for educational and research purposes. It does not provide investment recommendations or guarantee the accuracy of market data.

## Author

**itsbarz**

GitHub: [@itsbarz](https://github.com/itsbarz)