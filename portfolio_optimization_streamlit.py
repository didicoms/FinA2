import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Função para baixar os dados de ações
def get_data(tickers, start_date, end_date):
    st.write(f"Baixando dados de {start_date} até {end_date}...")
    all_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    all_data = all_data.ffill().bfill()  # Limpeza de dados (Forward Fill + Backward Fill)
    return all_data

# Função para calcular retorno anualizado e volatilidade
def calc_risk_return(prices):
    # Calculando retornos diários e, em seguida, anualizados
    daily_returns = prices.pct_change()
    annualized_return = daily_returns.mean() * 252  # 252 dias úteis por ano
    volatility = daily_returns.std() * np.sqrt(252)
    return annualized_return, volatility

# Função para otimizar o portfólio (exemplo simples)
def optimize_portfolio(prices):
    annualized_return, volatility = calc_risk_return(prices)

    # Simulando pesos para 10.000 portfólios
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(len(prices.columns))
        weights /= np.sum(weights)  # Garantir que a soma dos pesos seja 1
        portfolio_return = np.sum(weights * annualized_return)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(prices.cov() * 252, weights)))
        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = portfolio_return / portfolio_volatility  # Sharpe Ratio

    return results

# Função para plotar os resultados
def plot_results(results):
    plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatilidade')
    plt.ylabel('Retorno Anualizado')
    plt.title('Portfólios Otimizados')
    plt.show()

# Streamlit interface
st.title('Otimização de Portfólio com Streamlit')

# Seleção de ativos
tickers = st.multiselect(
    "Escolha os ativos", 
    ["ITUB4.SA", "BPAC11.SA", "ROXO34.SA", "XPBR31.SA", "VALE3.SA", "GGBR4.SA", "PETR4.SA", "SBSP3.SA", 
     "EQTL3.SA", "CPLE6.SA", "NEOE3.SA", "JHSF3.SA", "CYRE3.SA", "RAIL3.SA", "MULT3.SA", "IGTI11.SA", "ALOS3.SA", 
     "ABEV3.SA", "MRFG3.SA", "ASAI3.SA"],
    default=["ITUB4.SA", "VALE3.SA"]
)

# Definir a janela de datas
start_date = st.date_input("Data de Início", datetime.now() - timedelta(days=3*365))
end_date = st.date_input("Data de Fim", datetime.now())

# Baixar os dados e mostrar gráficos
if st.button('Carregar Dados'):
    if len(tickers) > 0:
        data = get_data(tickers, start_date, end_date)
        st.write(data.tail())  # Exibe as últimas 5 linhas dos dados

        # Calcular os retornos e volatilidade
        annualized_return, volatility = calc_risk_return(data)
        st.write("Retorno Anualizado (por ativo):", annualized_return)
        st.write("Volatilidade (por ativo):", volatility)

        # Otimizar o portfólio
        results = optimize_portfolio(data)
        plot_results(results)

        # Mostrar resultados do melhor portfólio
        st.write("Melhor Portfólio:")
        st.write("Retorno: ", results[0, results[2].argmax()])
        st.write("Volatilidade: ", results[1, results[2].argmax()])
        st.write("Sharpe Ratio: ", results[2].max())
