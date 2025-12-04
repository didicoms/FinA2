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

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="Lab Finanças - Victor Hugo", layout="wide")
np.random.seed(42) # Garante resultados idênticos

st.title("Lab Finanças: Escolha de Portfólio")
st.markdown("**Aluno:** Victor Hugo Lemos")

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros do Investidor")
investment_amount = st.sidebar.number_input("Valor a Investir (US$)", min_value=100.0, value=10000.0, step=100.0)
risk_free_annual = st.sidebar.number_input("Taxa Livre de Risco Anual (%)", value=4.0, step=0.1) / 100
test_days = st.sidebar.number_input("Dias de Backtest (Out-of-Sample)", value=252, step=1)
periodo_download = st.sidebar.selectbox("Período de Dados Históricos", ["2y", "5y", "10y"], index=1)

# Ativos
mag7 = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA"]
us_indices = ["SPY","QQQ","IWM"]
intl = ["VXUS","IEFA","EEM","ACWX"]
sectors = ["XLV","XLF","XLE","XLK","XLY"]
bonds = ["TLT","LQD","HYG"]
alts = ["GLD","VNQ","DBC"]
TICKERS = list(set(mag7 + us_indices + intl + sectors + bonds + alts))

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
    bounds = [(0.0, 1.0)] * n
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

# --- EXECUÇÃO ---
if st.sidebar.button("Rodar Análise Completa"):
    with st.spinner('Processando dados...'):
        
        # 1. Dados e Split
        prices = load_data(TICKERS, periodo_download)
        if len(prices) < test_days * 2:
            st.error("Dados insuficientes.")
            st.stop()
            
        train_prices = prices.iloc[:-test_days]
        test_prices = prices.iloc[-test_days:]
        split_date = train_prices.index[-1]
        
        train_rets = train_prices.pct_change().dropna()
        test_rets = test_prices.pct_change().dropna()
        
        # 2. Métricas Treino
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
        
        # 3. Processamento das Técnicas
        # K-Means Prep
        scaler = StandardScaler()
        X = scaler.fit_transform(metrics[["Ret","Vol","Sharpe","RSI","BB"]].fillna(0))
        
        # Lógica de seleção do melhor K (Silhouette)
        sil_scores_check = []
        k_values = range(2, 11)
        for k in k_values:
            km_temp = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels_temp = km_temp.fit_predict(X)
            sil_scores_check.append(silhouette_score(X, labels_temp))
        
        best_k = k_values[np.argmax(sil_scores_check)]
        
        # Rodar Final
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
        
        # Markowitz
        top_10 = metrics.sort_values("Sharpe", ascending=False).head(10)["Ticker"].tolist()
        mu_sub = train_rets[top_10].mean() * 252
        cov_sub = train_rets[top_10].cov() * 252
        w_opt = solve_max_sharpe(mu_sub.values, cov_sub.values, risk_free_annual)
        df_weights = pd.DataFrame({"Ticker": top_10, "Peso": w_opt}).sort_values("Peso", ascending=False).head(5)
        df_weights["Peso"] /= df_weights["Peso"].sum()
        sel_b = df_weights["Ticker"].tolist()
        weights_b = df_weights["Peso"].values
        
        # 4. Cálculo de Curvas
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

        # --- VISUALIZAÇÃO (5 ABAS) ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "1. Dados", "2. Técnica A", "3. Técnica B", "4. Backtest (Gráficos)", "5. Resultado Financeiro"
        ])
        
        with tab1:
            st.markdown("### Universo de Ativos e Dados de Treino")
            st.dataframe(metrics.set_index("Ticker").style.format("{:.2f}"))

        with tab2:
            st.markdown("### Definição Matemática dos Clusters")
            st.write("O algoritmo testa agrupar os ativos em 2 até 10 grupos. Escolhemos o número que maximiza a coesão (Silhouette).")
            
            # Recalcular inércias para plotagem (para exibir os gráficos)
            inertias = []
            sil_scores = []
            k_range = range(2, 11)
            for k in k_range:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                lb = km.fit_predict(X)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X, lb))
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**1. Método do Cotovelo (Inércia)**")
                fig_el, ax_el = plt.subplots(figsize=(5,3))
                ax_el.plot(k_range, inertias, marker='o')
                ax_el.set_xlabel("k"); ax_el.set_ylabel("Inércia")
                ax_el.grid(True, alpha=0.3)
                st.pyplot(fig_el)
            
            with c2:
                st.write("**2. Silhouette Score (Coesão)**")
                fig_sil, ax_sil = plt.subplots(figsize=(5,3))
                ax_sil.plot(k_range, sil_scores, marker='o', color='green')
                ax_sil.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor k={best_k}')
                ax_sil.legend()
                ax_sil.grid(True, alpha=0.3)
                st.pyplot(fig_sil)

            st.markdown("---")
            st.markdown(f"### Resultado: k={best_k} Clusters")
            st.success(f"Ativos Selecionados (Maior Sharpe de cada cluster): {', '.join(sel_a)}")
            
            fig, ax = plt.subplots()
            scatter = ax.scatter(metrics["Vol"], metrics["Ret"], c=metrics["Cluster"], cmap="viridis")
            plt.colorbar(scatter, label="Cluster ID")
            ax.set_xlabel("Volatilidade"); ax.set_ylabel("Retorno")
            ax.set_title("Mapa Final dos Clusters")
            st.pyplot(fig)

        with tab3:
            st.markdown("### Markowitz (Max Sharpe)")
            st.success(f"Carteira: {', '.join(sel_b)}")
            st.bar_chart(df_weights.set_index("Ticker"))

        with tab4:
            st.markdown("### Validação Técnica (Backtest)")
            st.write(f"Divisão Treino/Teste: **{split_date.date()}**")
            
            st.write("#### 1. Zoom no Período de Teste (Out-of-Sample)")
            fig_zoom, ax_z = plt.subplots(figsize=(10, 4))
            ax_z.plot(cum_a.index, cum_a, label="Cluster")
            ax_z.plot(cum_b.index, cum_b, label="Markowitz")
            ax_z.plot(cum_bench.index, cum_bench, label="Benchmark", linestyle="--", color="gray")
            ax_z.legend(); ax_z.grid(True, alpha=0.3)
            st.pyplot(fig_zoom)
            
            st.write("#### 2. Histórico Completo (Linha do Tempo)")
            fig_full, ax_f = plt.subplots(figsize=(10, 4))
            ax_f.plot(full_cum_a.index, full_cum_a, label="Cluster")
            ax_f.plot(full_cum_b.index, full_cum_b, label="Markowitz")
            ax_f.plot(full_cum_bench.index, full_cum_bench, label="Benchmark", linestyle="--", color="gray", alpha=0.5)
            ax_f.axvline(x=split_date, color='red', linestyle=':', label="Divisão")
            ax_f.legend(); ax_f.grid(True, alpha=0.3)
            st.pyplot(fig_full)

        with tab5:
            st.markdown(f"### Resultado para o Investidor (US$ {investment_amount:,.2f})")
            
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
                st.metric("Saldo Final", f"${final_a:,.2f}", f"{lucro_a:,.2f}")
                st.metric("Max Drawdown", f"{dd_a*100:.2f}%")
            with c2:
                st.info("**Técnica B (Markowitz)**")
                st.metric("Saldo Final", f"${final_b:,.2f}", f"{lucro_b:,.2f}")
                st.metric("Max Drawdown", f"{dd_b*100:.2f}%")
            with c3:
                st.warning("**Benchmark**")
                st.metric("Saldo Final", f"${final_bench:,.2f}")
                st.metric("Max Drawdown", f"{dd_bench*100:.2f}%")

else:
    st.info("👆 Clique no botão 'Rodar Análise Completa' para iniciar.")
