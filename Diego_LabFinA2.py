import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Lab Finanças - Diego Menezes", layout="wide")
np.random.seed(42)

st.title("Lab Finanças: Construção de Portfólio Otimizado")
st.markdown("**Aluno:** Diego Menezes")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros do Investidor")
investment_amount = st.sidebar.number_input("Valor a Investir (R$)", min_value=100.0, value=10000.0, step=100.0)
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=10.75, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# --- DEFINIÇÃO DOS ATIVOS (Pool de 20 Ativos) ---
# Corrigido MRFG3 conforme seu texto
TICKERS = [
    "ITUB4.SA", "BPAC11.SA", "ROXO34.SA", "XPBR31.SA", # Financeiro
    "PETR4.SA", "VALE3.SA", "GGBR4.SA",                # Commodities
    "SBSP3.SA", "EQTL3.SA", "CPLE6.SA", "NEOE3.SA", "RAIL3.SA", # Utilities
    "JHSF3.SA", "CYRE3.SA", "MULT3.SA", "IGTI11.SA", "ALOS3.SA", # Real Estate
    "ABEV3.SA", "ASAI3.SA", "MRFG3.SA"                 # Varejo/Consumo
]

# --- FUNÇÕES ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_width(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return (upper - lower) / sma

def annualize_return(daily_returns):
    return (1 + daily_returns.mean())**252 - 1

def annualize_vol(daily_returns):
    return daily_returns.std() * np.sqrt(252)

def sharpe_ratio_annual(daily_returns, rf):
    rf_daily = (1 + rf)**(1/252) - 1
    excess = daily_returns - rf_daily
    return (excess.mean() / (daily_returns.std() + 1e-12)) * np.sqrt(252)

def calculate_max_drawdown(daily_returns):
    cum_rets = (1 + daily_returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    return drawdown.min()

def solve_max_sharpe(mu, cov, rf):
    n = len(mu)
    def neg_sharpe(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w, np.dot(cov, w)))
        return -((ret - rf) / (vol + 1e-12))
    
    w0 = np.ones(n)/n
    # ALTERAÇÃO: Limite máximo de 30% por ativo conforme metodologia
    bounds = [(0.0, 0.30)] * n 
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
    return res.x

@st.cache_data
def load_data(tickers, period):
    data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    try:
        prices = data['Close']
    except KeyError:
        if isinstance(data.columns, pd.MultiIndex):
            prices = data.xs('Close', axis=1, level=0, drop_level=True)
        else:
            prices = data
    prices = prices.dropna(axis=1, how='all').ffill().bfill()
    return prices

# --- VISUALIZAÇÃO DA INTRODUÇÃO (Texto Personalizado) ---
def show_intro():
    st.markdown("""
    ## **1. Introdução e Justificativa dos Ativos**

    O objetivo deste trabalho é construir uma carteira de investimentos otimizada contendo 5 ativos, selecionados a partir de um universo inicial de 20 empresas.

    ### **Critério de Escolha do Universo (Pool de 20 ativos)**
    Para garantir que o algoritmo de otimização tenha insumos de qualidade, o *pool* inicial foi constituído buscando **diversificação setorial** e **representatividade econômica**. A seleção manual prévia evita o risco de concentração excessiva em um único fator macroeconômico.

    Os ativos foram divididos nos seguintes macro-setores:

    * **Setor Financeiro (Bancos e Fintechs):**
        * `ITUB4.SA` (Itaú), `BPAC11.SA` (BTG Pactual), `ROXO34.SA` (Nubank - BDR) e `XPBR31.SA` (XP Inc. - BDR).
        * **Justificativa:** Representam o crédito e serviços financeiros. A escolha mescla a solidez e dividendos dos bancos tradicionais com o potencial de crescimento das plataformas digitais.

    * **Commodities e Materiais Básicos:**
        * `VALE3.SA` (Minério), `GGBR4.SA` (Siderurgia) e `PETR4.SA` (Petróleo).
        * **Justificativa:** Ativos essenciais para capturar ciclos econômicos globais e oferecer proteção cambial implícita (receitas dolarizadas).

    * **Utilities e Infraestrutura:**
        * `SBSP3.SA` (Sabesp), `EQTL3.SA` (Equatorial), `CPLE6.SA` (Copel), `NEOE3.SA` (Neoenergia) e `RAIL3.SA` (Rumo).
        * **Justificativa:** Setores defensivos com receitas previsíveis e contratos ajustados pela inflação (IPCA/IGP-M).

    * **Real Estate (Construção e Shoppings):**
        * `CYRE3.SA` (Cyrela), `JHSF3.SA` (Alta Renda), `MULT3.SA` (Multiplan), `IGTI11.SA` (Iguatemi) e `ALOS3.SA` (Allos).
        * **Justificativa:** Ativos sensíveis à curva de juros, oferecendo potencial de valorização em ciclos de queda da Selic.

    * **Consumo e Varejo:**
        * `ABEV3.SA` (Ambev), `MRFG3.SA` (Marfrig) e `ASAI3.SA` (Assaí).
        * **Justificativa:** Foco em itens essenciais e resilientes: bebidas, proteínas e atacarejo.

    ---
    ### **Metodologia Aplicada**

    1.  **Clusterização (Machine Learning):** Utilização do algoritmo *K-Means* para agrupar os ativos com base em risco (Volatilidade) e retorno.
    2.  **Stock Picking Quantitativo:** Seleção do melhor ativo de cada cluster para garantir diversificação estrutural.
    3.  **Otimização de Markowitz:** Definição dos pesos ideais com **restrição máxima de 30% por ativo** para evitar concentração.
    4.  **Backtesting (Walk-Forward):** Validação da estratégia "fora da amostra" comparando com o Benchmark (Média do Universo).
    """)

# --- EXECUÇÃO PRINCIPAL ---
if st.sidebar.button("Rodar Análise Completa"):
    with st.spinner('Baixando dados e processando modelos...'):
        
        prices = load_data(TICKERS, periodo_download)
        
        if prices.empty or len(prices) < test_days * 2:
            st.error("Dados insuficientes.")
            st.stop()
            
        train_prices = prices.iloc[:-test_days]
        test_prices = prices.iloc[-test_days:]
        split_date = train_prices.index[-1]
        
        train_rets = train_prices.pct_change().dropna()
        test_rets = test_prices.pct_change().dropna()
        
        # Cálculo de Métricas
        summary = []
        for t in train_prices.columns:
            r = train_rets[t]
            p = train_prices[t]
            try:
                curr_rsi = calculate_rsi(p).iloc[-1]
                curr_bb = calculate_bollinger_width(p).iloc[-1]
            except:
                curr_rsi, curr_bb = 50, 0
            summary.append([t, annualize_return(r), annualize_vol(r), sharpe_ratio_annual(r, risk_free_annual), curr_rsi, curr_bb])
            
        metrics = pd.DataFrame(summary, columns=["Ticker","Ret","Vol","Sharpe","RSI","BB"])
        
        # Clusterização
        scaler = StandardScaler()
        X = scaler.fit_transform(metrics[["Ret","Vol","Sharpe","RSI","BB"]].fillna(0))
        
        sil_scores_check = []
        k_values = range(2, 11)
        for k in k_values:
            km_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
            lb = km_temp.fit_predict(X)
            sil_scores_check.append(silhouette_score(X, lb))
        
        best_k = k_values[np.argmax(sil_scores_check)]
        kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=42)
        metrics["Cluster"] = kmeans.fit_predict(X)
        
        sel_a = []
        for c in sorted(metrics["Cluster"].unique()):
            top = metrics[metrics["Cluster"]==c].sort_values("Sharpe", ascending=False).iloc[0]["Ticker"]
            sel_a.append(top)
        
        if len(sel_a) < 5:
            rest = metrics[~metrics["Ticker"].isin(sel_a)].sort_values("Sharpe", ascending=False)
            sel_a.extend(rest["Ticker"].head(5 - len(sel_a)).tolist())
        sel_a = sel_a[:5]
        
        # Markowitz (com restrição de 30%)
        top_10 = metrics.sort_values("Sharpe", ascending=False).head(10)["Ticker"].tolist()
        mu_sub = train_rets[top_10].mean() * 252
        cov_sub = train_rets[top_10].cov() * 252
        w_opt = solve_max_sharpe(mu_sub.values, cov_sub.values, risk_free_annual)
        
        df_weights = pd.DataFrame({"Ticker": top_10, "Peso": w_opt}).sort_values("Peso", ascending=False).head(5)
        df_weights["Peso"] /= df_weights["Peso"].sum() # Renormaliza para os top 5
        sel_b = df_weights["Ticker"].tolist()
        weights_b = df_weights["Peso"].values
        
        # Curvas
        r_test_a = test_rets[sel_a].mean(axis=1).fillna(0)
        r_test_b = test_rets[sel_b].mul(weights_b, axis=1).sum(axis=1).fillna(0)
        r_test_bench = test_rets.mean(axis=1).fillna(0)
        
        cum_a = (1 + r_test_a).cumprod()
        cum_b = (1 + r_test_b).cumprod()
        cum_bench = (1 + r_test_bench).cumprod()
        
        full_ret_a = prices[sel_a].pct_change().mean(axis=1).fillna(0)
        full_ret_b = prices[sel_b].pct_change().mul(weights_b, axis=1).sum(axis=1).fillna(0)
        full_ret_bench = prices.pct_change().mean(axis=1).fillna(0)
        
        full_cum_a = (1 + full_ret_a).cumprod()
        full_cum_b = (1 + full_ret_b).cumprod()
        full_cum_bench = (1 + full_ret_bench).cumprod()

        # --- VISUALIZAÇÃO DE ABAS ---
        # Adicionei a aba "Introdução" como a primeira
        tab_intro, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1. Introdução & Ativos", "2. Dados", "3. Correlação", "4. Técnica A (Cluster)", "5. Técnica B (Markowitz)", "6. Backtest", "7. Resultado"
        ])
        
        with tab_intro:
            show_intro()

        with tab1:
            st.markdown("### Universo de Ativos e Dados de Treino")
            st.dataframe(metrics.set_index("Ticker").style.format("{:.2f}"))

        with tab2:
            st.markdown("### Matriz de Correlação (Período de Treino)")
            corr_matrix = train_rets.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).format("{:.2f}"))

        with tab3:
            st.markdown(f"### Resultado: k={best_k} Clusters")
            st.success(f"Ativos Selecionados (Cluster): {', '.join(sel_a)}")
            
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                scatter = ax.scatter(metrics["Vol"], metrics["Ret"], c=metrics["Cluster"], cmap="viridis")
                plt.colorbar(scatter, label="Cluster ID")
                ax.set_xlabel("Volatilidade"); ax.set_ylabel("Retorno")
                st.pyplot(fig)
            with c2:
                # Gráfico de barras dos ativos selecionados
                st.bar_chart(metrics[metrics["Ticker"].isin(sel_a)].set_index("Ticker")["Sharpe"])

        with tab4:
            st.markdown("### Markowitz (Max Sharpe - Limite 30%)")
            st.success(f"Carteira Otimizada: {', '.join(sel_b)}")
            st.bar_chart(df_weights.set_index("Ticker"))

        with tab5:
            st.markdown("### Validação Técnica (Backtest)")
            st.write(f"Divisão Treino/Teste: **{split_date.date()}**")
            fig_zoom, ax_z = plt.subplots(figsize=(10, 4))
            ax_z.plot(cum_a.index, cum_a, label="Cluster")
            ax_z.plot(cum_b.index, cum_b, label="Markowitz")
            ax_z.plot(cum_bench.index, cum_bench, label="Benchmark", linestyle="--", color="gray")
            ax_z.legend(); ax_z.grid(True, alpha=0.3)
            st.pyplot(fig_zoom)

        with tab6:
            st.markdown(f"### Resultado para o Investidor (R$ {investment_amount:,.2f})")
            final_a = investment_amount * cum_a.iloc[-1]
            final_b = investment_amount * cum_b.iloc[-1]
            final_bench = investment_amount * cum_bench.iloc[-1]
            
            lucro_a = final_a - investment_amount
            lucro_b = final_b - investment_amount
            
            dd_a = calculate_max_drawdown(r_test_a)
            dd_b = calculate_max_drawdown(r_test_b)
            dd_bench = calculate_max_drawdown(r_test_bench)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.info("**Técnica A (Cluster)**")
                st.metric("Saldo Final", f"R$ {final_a:,.2f}", f"{lucro_a:,.2f}")
                st.metric("Max Drawdown", f"{dd_a*100:.2f}%")
            with c2:
                st.info("**Técnica B (Markowitz)**")
                st.metric("Saldo Final", f"R$ {final_b:,.2f}", f"{lucro_b:,.2f}")
                st.metric("Max Drawdown", f"{dd_b*100:.2f}%")
            with c3:
                st.warning("**Benchmark**")
                st.metric("Saldo Final", f"R$ {final_bench:,.2f}")
                st.metric("Max Drawdown", f"{dd_bench*100:.2f}%")

else:
    # Mostra a introdução na tela inicial antes de rodar
    show_intro()
    st.info("👆 Clique no botão 'Rodar Análise Completa' na barra lateral para iniciar os cálculos.")
