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

test_days = st.sidebar.number_input(
    "Dias de Backtest (Out-of-Sample)", 
    value=252, 
    step=1,
    help="Dias úteis separados para validar o modelo. Ex: 252 dias ≈ 1 ano útil."
)

anos_historico = st.sidebar.slider("Anos de Dados Históricos", min_value=2, max_value=10, value=5)
periodo_download = f"{anos_historico}y"

# --- DEFINIÇÃO DOS ATIVOS ---
TICKERS = [
    "ITUB4.SA", "BPAC11.SA", "ROXO34.SA", "XPBR31.SA", 
    "PETR4.SA", "VALE3.SA", "GGBR4.SA",                
    "SBSP3.SA", "EQTL3.SA", "CPLE6.SA", "NEOE3.SA", "RAIL3.SA", 
    "JHSF3.SA", "CYRE3.SA", "MULT3.SA", "IGTI11.SA", "ALOS3.SA", 
    "ABEV3.SA", "ASAI3.SA", "MRFG3.SA"                 
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
    bounds = [(0.05, 1.0)] * n 
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

def show_intro():
    st.markdown("""
    ## **1. Introdução e Justificativa dos Ativos**
    O objetivo deste trabalho é construir uma carteira de investimentos otimizada contendo 5 ativos, selecionados a partir de um universo inicial de 20 empresas.
    
    ### **Critério de Escolha do Universo**
    O *pool* inicial foi constituído buscando **diversificação setorial** e **representatividade econômica**.
    * **Financeiro:** `ITUB4`, `BPAC11`, `ROXO34`, `XPBR31`
    * **Commodities:** `VALE3`, `GGBR4`, `PETR4`
    * **Utilities:** `SBSP3`, `EQTL3`, `CPLE6`, `NEOE3`, `RAIL3`
    * **Real Estate:** `CYRE3`, `JHSF3`, `MULT3`, `IGTI11`, `ALOS3`
    * **Varejo:** `ABEV3`, `MRFG3`, `ASAI3`

    ---
    ### **Metodologia Aplicada**
    1.  **Clusterização (K-Means):** Agrupamento por perfil de risco/retorno.
    2.  **Otimização de Markowitz:** Maximização do Sharpe Ratio (Min. 5% por ativo).
    3.  **Simulação de Monte Carlo:** Geração de cenários aleatórios para visualizar a Fronteira Eficiente.
    4.  **Backtesting:** Validação "Out-of-Sample".
    """)

# --- EXECUÇÃO PRINCIPAL ---
if st.sidebar.button("Rodar Otimização"):
    with st.spinner(f'Baixando {anos_historico} anos de dados e processando...'):
        
        prices = load_data(TICKERS, periodo_download)
        
        if prices.empty or len(prices) < test_days + 126:
            st.error(f"Dados insuficientes. Tente aumentar os anos na barra lateral.")
            st.stop()
            
        train_prices = prices.iloc[:-test_days]
        test_prices = prices.iloc[-test_days:]
        
        train_rets = train_prices.pct_change().dropna()
        test_rets = test_prices.pct_change().dropna()
        
        # Cálculo de Métricas Iniciais
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
        
        # 1. Clusterização
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
        
        # 2. Markowitz (Top 10)
        top_10 = metrics.sort_values("Sharpe", ascending=False).head(10)["Ticker"].tolist()
        mu_sub = train_rets[top_10].mean() * 252
        cov_sub = train_rets[top_10].cov() * 252
        w_opt = solve_max_sharpe(mu_sub.values, cov_sub.values, risk_free_annual)
        
        df_weights = pd.DataFrame({"Ticker": top_10, "Peso": w_opt}).sort_values("Peso", ascending=False).head(5)
        df_weights["Peso"] /= df_weights["Peso"].sum()
        sel_b = df_weights["Ticker"].tolist()
        weights_b = df_weights["Peso"].values
        
        # 3. Monte Carlo (NOVO DIFERENCIAL)
        # Gera 2000 portfólios aleatórios com os ativos do Top 10 para desenhar a fronteira
        num_simulations = 2000
        sim_results = np.zeros((3, num_simulations))
        
        for i in range(num_simulations):
            w = np.random.random(len(top_10))
            w /= np.sum(w)
            p_ret = np.sum(w * mu_sub.values)
            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_sub.values, w)))
            sim_results[0,i] = p_ret
            sim_results[1,i] = p_vol
            sim_results[2,i] = (p_ret - risk_free_annual) / p_vol # Sharpe

        # Métricas do Markowitz Otimizado (para plotar a estrela)
        opt_ret = np.sum(w_opt * mu_sub.values)
        opt_vol = np.sqrt(np.dot(w_opt.T, np.dot(cov_sub.values, w_opt)))
        
        # 4. Curvas Backtest
        r_test_a = test_rets[sel_a].mean(axis=1).fillna(0)
        r_test_b = test_rets[sel_b].mul(weights_b, axis=1).sum(axis=1).fillna(0)
        r_test_bench = test_rets.mean(axis=1).fillna(0)
        
        cum_a = (1 + r_test_a).cumprod()
        cum_b = (1 + r_test_b).cumprod()
        cum_bench = (1 + r_test_bench).cumprod()
        
        # --- VISUALIZAÇÃO ---
        tab_intro, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1. Introdução", "2. Dados", "3. Correlação", "4. Clusterização", "5. Markowitz & Fronteira", "6. Backtest", "7. Resultado Final"
        ])
        
        with tab_intro:
            show_intro()

        with tab1:
            st.markdown("### Universo de Ativos e Dados de Treino")
            st.dataframe(metrics.set_index("Ticker").style.format("{:.2f}"))

        with tab2:
            st.markdown("### Matriz de Correlação")
            corr_matrix = train_rets.corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1).format("{:.2f}"))

        with tab3:
            st.markdown(f"### Resultado: k={best_k} Clusters")
            st.success(f"Ativos Selecionados: {', '.join(sel_a)}")
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots()
                scatter = ax.scatter(metrics["Vol"], metrics["Ret"], c=metrics["Cluster"], cmap="viridis")
                plt.colorbar(scatter, label="Cluster ID")
                ax.set_xlabel("Volatilidade"); ax.set_ylabel("Retorno")
                st.pyplot(fig)
            with c2:
                st.bar_chart(metrics[metrics["Ticker"].isin(sel_a)].set_index("Ticker")["Sharpe"])

        with tab4:
            st.markdown("### Markowitz e Fronteira Eficiente (Simulação de Monte Carlo)")
            st.info("O gráfico abaixo mostra 2.000 simulações de portfólios aleatórios (pontos cinza). A estrela vermelha indica a carteira otimizada matemática.")
            
            col_graph, col_weights = st.columns([2, 1])
            
            with col_graph:
                fig_mc, ax_mc = plt.subplots(figsize=(8,5))
                sc_mc = ax_mc.scatter(sim_results[1,:], sim_results[0,:], c=sim_results[2,:], cmap='viridis', alpha=0.3, s=10)
                plt.colorbar(sc_mc, label='Sharpe Ratio')
                # Plota o ponto otimizado
                ax_mc.scatter(opt_vol, opt_ret, c='red', s=200, marker='*', label='Portfólio Otimizado')
                ax_mc.set_xlabel('Volatilidade (Risco)')
                ax_mc.set_ylabel('Retorno Esperado')
                ax_mc.set_title('Fronteira Eficiente')
                ax_mc.legend()
                st.pyplot(fig_mc)
            
            with col_weights:
                st.write("**Pesos Otimizados:**")
                st.dataframe(df_weights.set_index("Ticker").style.format("{:.1%}"))

        with tab5:
            st.markdown("### Validação Técnica (Backtest)")
            fig_zoom, ax_z = plt.subplots(figsize=(10, 4))
            ax_z.plot(cum_a.index, cum_a, label="Cluster")
            ax_z.plot(cum_b.index, cum_b, label="Markowitz")
            ax_z.plot(cum_bench.index, cum_bench, label="Benchmark", linestyle="--", color="gray")
            ax_z.legend(); ax_z.grid(True, alpha=0.3)
            st.pyplot(fig_zoom)

        with tab6:
            st.markdown(f"### Resultado Consolidado (R$ {investment_amount:,.2f})")
            
            final_a = investment_amount * cum_a.iloc[-1]
            final_b = investment_amount * cum_b.iloc[-1]
            final_bench = investment_amount * cum_bench.iloc[-1]
            
            # Tabela Resumo (NOVO DIFERENCIAL)
            summary_data = {
                "Estratégia": ["Técnica A (Cluster)", "Técnica B (Markowitz)", "Benchmark"],
                "Saldo Final (R$)": [final_a, final_b, final_bench],
                "Retorno Total (%)": [(cum_a.iloc[-1]-1)*100, (cum_b.iloc[-1]-1)*100, (cum_bench.iloc[-1]-1)*100],
                "Max Drawdown (%)": [calculate_max_drawdown(r_test_a)*100, calculate_max_drawdown(r_test_b)*100, calculate_max_drawdown(r_test_bench)*100]
            }
            df_summary = pd.DataFrame(summary_data).set_index("Estratégia")
            
            # Mostra a tabela bonita
            st.table(df_summary.style.format({
                "Saldo Final (R$)": "R$ {:,.2f}",
                "Retorno Total (%)": "{:+.2f}%",
                "Max Drawdown (%)": "{:.2f}%"
            }))

            # Cards visuais
            c1, c2, c3 = st.columns(3)
            c1.metric("Cluster (Saldo)", f"R$ {final_a:,.2f}", f"{summary_data['Retorno Total (%)'][0]:.2f}%")
            c2.metric("Markowitz (Saldo)", f"R$ {final_b:,.2f}", f"{summary_data['Retorno Total (%)'][1]:.2f}%")
            c3.metric("Benchmark (Saldo)", f"R$ {final_bench:,.2f}", f"{summary_data['Retorno Total (%)'][2]:.2f}%")

else:
    show_intro()
    st.info("Clique no botão 'Rodar Otimização' na barra lateral para iniciar os cálculos.")
