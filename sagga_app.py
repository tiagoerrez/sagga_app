import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t
import plotly.graph_objects as go

# Import from your package using the correct path
import crypto_toolkit as ct
import cryptocompare_toolkit as cct

import logging  # Add this import

# Add page config and logging setup right after imports
st.set_page_config(
    page_title="Crypto Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add custom CSS right before the main() function
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .reportview-container {
        background: #f0f2f6
    }
    </style>
    """, unsafe_allow_html=True)

# Add the validation function before main()
def validate_crypto_symbols(symbols):
    valid_symbols = []
    invalid_symbols = []
    for symbol in symbols:
        try:
            if symbol and len(symbol) > 0:
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append(symbol)
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {str(e)}")
            invalid_symbols.append(symbol)
    return valid_symbols, invalid_symbols

# Add caching decorator before the existing display_weights function
@st.cache_data(ttl=3600)
def fetch_crypto_data(coins, version='v3'):
    if version == 'v2':
        return cct.get_normal_returns_v2(coins)
    elif version == 'v3':
        return cct.get_normal_returns_v3(coins)
    else:
        raise ValueError("Unsupported version. Use 'v2' or 'v3'.")

def display_weights(weights, returns, method_name, version):
    weights_df = pd.DataFrame({'Cryptocurrency': weights.index, 'Weight': weights.values * 100})
    weights_df['Weight'] = weights_df['Weight'].map("{:.2f}%".format)
    
    # display data frames
    st.dataframe(weights_df)

    if version == 'v2':
        er = ct.annualize_rets(returns, 365)
        cov = returns.cov() * 365
        port_ret = ct.portfolio_return(weights, er)
        port_vol = ct.portfolio_vol(weights, cov)
        summary = ct.summary_stats(returns, riskfree_rate=0.03)
    elif version == 'v3':
        er = ct.annualize_rets_v3(returns, 365)
        cov = ct.cov_v3(returns).astype(float) * 365
        port_ret = ct.portfolio_return(weights, er)
        port_vol = ct.portfolio_vol(weights, cov)
        summary = pd.DataFrame(index=weights.index)
        for coin in returns:
            summary.loc[coin] = ct.summary_stats(returns[coin].to_frame(coin)).iloc[0]

    st.write(f"**{method_name} Portfolio Metrics**")
    st.write(f"Annualized Portfolio Return: {port_ret:.4f}")
    st.write(f"Annualized Portfolio Volatility: {port_vol:.4f}")
    st.dataframe(summary)

    fig = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values * 100, hole=0.4)])
    fig.update_layout(title=f'{method_name} Portfolio Allocation')
    st.plotly_chart(fig)

    csv = weights_df.to_csv(index=False)
    st.download_button(label=f"Download {method_name} Weights", data=csv, file_name=f"{method_name.lower()}_weights.csv", mime="text/csv")

def plot_efficient_frontier(er, cov, riskfree_rate):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    weights = ct.optimal_weights(50, er, cov)
    rets = [ct.portfolio_return(w, er) for w in weights]
    vols = [ct.portfolio_vol(w, cov) for w in weights]
    
    sns.scatterplot(x=vols, y=rets, ax=ax, color='cyan', label='Efficient Frontier', s=50)
    w_msr = ct.msr(riskfree_rate, er, cov)
    r_msr = ct.portfolio_return(w_msr, er)
    vol_msr = ct.portfolio_vol(w_msr, cov)
    ax.plot([0, vol_msr], [riskfree_rate, r_msr], color='green', linestyle='--', label='CML')
    ax.scatter([vol_msr], [r_msr], color='red', label='MSR', s=100)

    w_gmv = ct.gmv(cov)
    r_gmv = ct.portfolio_return(w_gmv, er)
    vol_gmv = ct.portfolio_vol(w_gmv, cov)
    ax.scatter([vol_gmv], [r_gmv], color='yellow', label='GMV', s=100)

    n = er.shape[0]
    w_ew = np.repeat(1/n, n)
    r_ew = ct.portfolio_return(w_ew, er)
    vol_ew = ct.portfolio_vol(w_ew, cov)
    ax.scatter([vol_ew], [r_ew], color='purple', label='Equal Weight', s=100)

    ax.set_xlabel('Volatility')
    ax.set_ylabel('Return')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def plot_correlation_matrix(returns, version):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = returns.corr() if version == 'v2' else ct.corr_v3(returns)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1, center=0)
    ax.set_title('Correlation Matrix')
    return fig

def plot_returns_distribution(returns, asset_name=None):
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    if isinstance(returns, pd.Series):
        returns.plot(ax=ax1, color='cyan')
    else:
        returns.mean(axis=1).plot(ax=ax1, color='cyan')
    ax1.set_title(f"{'Portfolio' if asset_name is None else asset_name} Returns")
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Returns')
    ax1.grid(True, linestyle='--', alpha=0.3)

    sns.histplot(returns.dropna(), bins=50, ax=ax2, color='cyan', stat='density', label='Actual')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax2.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', label='Normal')
    df = len(returns.dropna()) - 1
    ax2.plot(x, t.pdf(x, df, returns.mean(), returns.std()), 'g--', label='t-Student')
    ax2.set_title('Distribution')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.3)
    return fig

def plot_var_cvar(returns, rolling=False, window=30):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    if rolling:
        var = ct.var_gaussian(returns, level=5, window=window)
        cvar = ct.cvar_gaussian(returns, level=5, window=window)
        var.plot(ax=ax, label='Rolling Gaussian VaR (5%)', color='red')
        cvar.plot(ax=ax, label='Rolling Gaussian CVaR (5%)', color='orange')
    else:
        var = ct.var_gaussian(returns, level=5)
        cvar = ct.cvar_gaussian(returns, level=5)
        ax.axhline(var, color='red', label=f'Gaussian VaR (5%): {var:.4f}')
        ax.axhline(cvar, color='orange', label=f'Gaussian CVaR (5%): {cvar:.4f}')
    ax.set_title('Gaussian VaR and CVaR')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def plot_monte_carlo(returns, n_scenarios=100, n_years=1):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    mu = returns.mean() * 365
    sigma = returns.std() * np.sqrt(365)
    sim = ct.gbm(n_years=n_years, n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=365)
    sim.plot(ax=ax, legend=False, alpha=0.3, color='cyan')
    ax.set_title('Monte Carlo Simulation')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.3)
    return fig

def main():
    st.title("Crypto Portfolio Optimizer")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio Optimizer", "Efficient Frontier", "Analytics", "Performance", "About"])

    with tab1:
        st.header("Portfolio Optimization")
        crypto_input = st.text_area("Enter cryptocurrency symbols (comma-separated)", "BTC, ETH, XRP")
        risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
        version = st.selectbox("Returns Version", ['v2', 'v3'], index=1)
        custom_weights = st.text_input("Custom Weights (comma-separated, sum to 1)", "", help="e.g., 0.4,0.3,0.3")

        coins = [coin.strip().upper() for coin in crypto_input.split(",") if coin.strip()]
        if len(coins) < 2:
            st.error("Please enter at least 2 cryptocurrencies.")
        else:
            col1, col2, col3 = st.columns(3)
            returns = fetch_crypto_data(coins, version)

            with col1:
                if st.button("Calculate GMV Weights"):
                    with st.spinner("Calculating GMV weights..."):
                        if version == 'v3':
                            weights = ct.get_gmv_weights(coins, version='v3')
                        else:
                            er = ct.annualize_rets(returns, 365)
                            cov = returns.cov() * 365
                            weights = pd.Series(ct.gmv(cov), index=coins)
                        display_weights(weights, returns, "Global Minimum Variance", version)

            with col2:
                if st.button("Calculate MSR Weights"):
                    with st.spinner("Calculating MSR weights..."):
                        if version == 'v2':
                            er = ct.annualize_rets(returns, 365)
                            cov = returns.cov() * 365
                        else:
                            er = ct.annualize_rets_v3(returns, 365)
                            cov = ct.cov_v3(returns).astype(float) * 365
                        weights = pd.Series(ct.msr(risk_free_rate, er, cov), index=coins)
                        display_weights(weights, returns, "Maximum Sharpe Ratio", version)

            with col3:
                if custom_weights and st.button("Use Custom Weights"):
                    weights = np.array([float(w.strip()) for w in custom_weights.split(',')])
                    if len(weights) == len(coins) and abs(sum(weights) - 1) < 1e-6:
                        weights = pd.Series(weights, index=coins)
                        display_weights(weights, returns, "Custom", version)
                    else:
                        st.error("Weights must match number of coins and sum to 1.")

    with tab2:
        st.header("Efficient Frontier")
        if 'coins' in locals() and len(coins) >= 2:
            if version == 'v2':
                er = ct.annualize_rets(returns, 365)
                cov = returns.cov() * 365
            else:
                er = ct.annualize_rets_v3(returns, 365)
                cov = ct.cov_v3(returns).astype(float) * 365
            fig = plot_efficient_frontier(er, cov, risk_free_rate)
            st.pyplot(fig)

    with tab3:
        st.header("Analytics")
        if 'coins' in locals() and len(coins) >= 2:
            st.subheader("Correlation Matrix")
            fig = plot_correlation_matrix(returns, version)
            st.pyplot(fig)

            st.subheader("Returns Analysis")
            asset = st.selectbox("Select Asset or Portfolio", ['Portfolio'] + coins)
            rolling = st.checkbox("Show Rolling VaR/CVaR", False)
            window = st.slider("Rolling Window (days)", 10, 100, 30) if rolling else None

            if asset == 'Portfolio':
                port_returns = returns.mean(axis=1) if version == 'v2' else pd.concat([returns[coin] for coin in coins], axis=1).mean(axis=1)
            else:
                port_returns = returns[asset] if version == 'v3' else returns[asset]

            fig = plot_returns_distribution(port_returns, asset)
            st.pyplot(fig)
            fig = plot_var_cvar(port_returns, rolling, window)
            st.pyplot(fig)

            st.subheader("Monte Carlo Simulation")
            n_scenarios = st.slider("Number of Scenarios", 50, 500, 100)
            n_years = st.slider("Years", 1, 5, 1)
            fig = plot_monte_carlo(port_returns, n_scenarios, n_years)
            st.pyplot(fig)

    with tab4:
        st.header("Performance Metrics")
        if 'coins' in locals() and len(coins) >= 2:
            st.subheader("Portfolio Metrics")
            method = st.selectbox("Select Portfolio", ["GMV", "MSR", "Custom"])
            if method == "GMV":
                weights = ct.get_gmv_weights(coins, version='v3') if version == 'v3' else pd.Series(ct.gmv(returns.cov() * 365), index=coins)
            elif method == "MSR":
                er = ct.annualize_rets(returns, 365) if version == 'v2' else ct.annualize_rets_v3(returns, 365)
                cov = returns.cov() * 365 if version == 'v2' else ct.cov_v3(returns).astype(float) * 365
                weights = pd.Series(ct.msr(risk_free_rate, er, cov), index=coins)
            else:
                weights = pd.Series([float(w.strip()) for w in custom_weights.split(',')], index=coins) if custom_weights else None
            
            if weights is not None:
                if version == 'v2':
                    port_returns = (returns @ weights).to_frame('Portfolio')
                else:
                    port_returns = pd.concat([returns[coin] * weights[coin] for coin in returns], axis=1).sum(axis=1).to_frame('Portfolio')
                metrics = ct.summary_stats(port_returns, riskfree_rate=risk_free_rate)
                st.dataframe(metrics)

    with tab5:
        st.header("About Portfolio Optimization")
        st.write("""
        This dashboard implements:
        - **GMV**: Minimizes risk.
        - **MSR**: Maximizes risk-adjusted return.
        - **Custom Weights**: User-defined allocations.
        - **Analytics**: Correlation, distribution, VaR/CVaR, Monte Carlo.
        - **Performance**: Key portfolio metrics.
        """)

if __name__ == "__main__":
    main()