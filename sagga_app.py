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
def fetch_crypto_data(coins, version='v3', clean_outliers=False, z_threshold=5):
    if version == 'v2':
        returns = cct.get_normal_returns_v2(coins)
        if clean_outliers:
            returns = ct.clean_dataframe(returns, z_threshold=z_threshold)
    elif version == 'v3':
        returns = cct.get_normal_returns_v3(coins)
        if clean_outliers:
            returns = ct.clean_dict(returns, z_threshold=z_threshold)
    else:
        raise ValueError("Unsupported version. Use 'v2' or 'v3'.")
    return returns

def display_weights(weights, returns, method_name, version, risk_free_rate):
    weights_df = pd.DataFrame({'Cryptocurrency': weights.index, 'Weight': weights.values * 100})
    weights_df['Weight'] = weights_df['Weight'].map("{:.2f}%".format)
    
    # display data frames
    st.dataframe(weights_df)

    if version == 'v2':
        er = ct.annualize_rets(returns, 365)
        cov = returns.cov() * 365
        port_ret = ct.portfolio_return(weights, er)
        port_vol = ct.portfolio_vol(weights, cov)
        summary = ct.summary_stats(returns, riskfree_rate=risk_free_rate)
    elif version == 'v3':
        er = ct.annualize_rets_v3(returns, 365)
        cov = ct.cov_v3(returns).astype(float) * 365
        port_ret = ct.portfolio_return(weights, er)
        port_vol = ct.portfolio_vol(weights, cov)
        # Summary stats for individual coins
        summary_data = {}
        for coin in returns:
            stats = ct.summary_stats(returns[coin].to_frame(coin), riskfree_rate=0.03)
            summary_data[coin] = stats.iloc[0]
        summary = pd.DataFrame(summary_data).T

    st.write(f"**{method_name} Portfolio Metrics**")
    st.write(f"Annualized Portfolio Return: {port_ret:.4f}")
    st.write(f"Annualized Portfolio Volatility: {port_vol:.4f}")
    st.dataframe(summary)

    fig = go.Figure(data=[go.Pie(labels=weights.index, values=weights.values * 100, hole=0.4)])
    fig.update_layout(title=f'{method_name} Portfolio Allocation')
    st.plotly_chart(fig)

    csv = weights_df.to_csv(index=False)
    st.download_button(label=f"Download {method_name} Weights", data=csv, file_name=f"{method_name.lower()}_weights.csv", mime="text/csv")

def plot_efficient_frontier(er, cov, riskfree_rate, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
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
    if log_scale:
        ax.set_yscale('log')
        ax.set_xscale('log')
    return fig

def plot_correlation_matrix(returns, version):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = returns.corr() if version == 'v2' else ct.corr_v3(returns)
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1, center=0)
    ax.set_title('Correlation Matrix')
    return fig

# Updated Distribution Plot (removed returns plot)
def plot_returns_distribution(returns, asset_name=None, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(returns, pd.Series):
        sns.histplot(returns.dropna(), bins=50, ax=ax, color='cyan', stat='density', label='Actual')
    else:
        sns.histplot(returns.mean(axis=1).dropna(), bins=50, ax=ax, color='cyan', stat='density', label='Actual')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', label='Normal')
    ax.plot(x, t.pdf(x, 5, returns.mean(), returns.std()), 'g--', label='t-Student (df=5)')
    ax.set_title(f"{'Portfolio' if asset_name is None else asset_name} Returns Distribution")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

def plot_var_cvar(returns, rolling=False, window=30, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)  # Portfolio returns if DataFrame
    
    returns.plot(ax=ax, color='cyan', alpha=0.5, label='Returns')
    if rolling:
        var = ct.var_historic(returns, level=5, window=window)  # Use historic VaR for simplicity
        cvar = ct.cvar_historic(returns, level=5, window=window)
        var.plot(ax=ax, label='Rolling VaR (5%)', color='red')
        cvar.plot(ax=ax, label='Rolling CVaR (5%)', color='orange')#
        current_var = var.iloc[-1]
        current_cvar = cvar.iloc[-1]
    else:
        var = ct.var_historic(returns, level=5)
        cvar = ct.cvar_historic(returns, level=5)
        ax.axhline(var, color='red', label=f'VaR (5%): {var:.4f}', linestyle='--')
        ax.axhline(cvar, color='orange', label=f'CVaR (5%): {cvar:.4f}', linestyle='--')
        current_var = var
        current_cvar = cvar
    ax.set_title(f'Returns vs VaR/CVaR - Current VaR: {current_var:.4f}, CVaR: {current_cvar:.4f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

def plot_monte_carlo(returns, n_scenarios=100, n_years=1, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    mu = returns.mean() * 365
    sigma = returns.std() * np.sqrt(365)
    sim = ct.gbm(n_years=n_years, n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=365)
    
    # Plot all simulations with no label to exclude from legend
    sim.plot(ax=ax, legend=False, alpha=0.3, color='cyan', label=None)

    # # Plot simulations manually to control legend
    # for i in range(n_scenarios):
    #     ax.plot(sim[:, i], color='cyan', alpha=0.3, label=None)  # No label for individual paths
    
    # Plot only 5th and 95th percentiles with explicit labels for legend
    ax.plot(sim.quantile(0.05, axis=1), color='red', label='5th Percentile')
    ax.plot(sim.quantile(0.95, axis=1), color='green', label='95th Percentile')
    
    ax.set_title('Monte Carlo Simulation with Confidence Intervals')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    # ax.legend()  # Only 5th and 95th percentiles will appear in the legend
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

def plot_volatility(returns, asset_name=None, window=30, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(returns, pd.Series):
        vol = returns.rolling(window).std() # * np.sqrt(365)  # Annualized volatility
        vol.plot(ax=ax, color='cyan', label=f'{asset_name} Volatility')
        current_vol = vol.iloc[-1]
    else:
        vol = returns.mean(axis=1).rolling(window).std() # * np.sqrt(365)
        vol.plot(ax=ax, color='cyan', label='Portfolio Volatility')
        current_vol = vol.iloc[-1]
    ax.set_title(f"{'Portfolio' if asset_name is None else asset_name} Rolling Volatility (window={window}) - Current: {current_vol:.4f}")
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{window}-day Volatility')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

# New Price Chart
# Updated plot_price_chart to use pre-fetched data
def plot_price_chart(historical_data, coins, currency='USD', selected_coin=None, start_date=None, end_date=None, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    try:
        if selected_coin is None:
            for coin in coins:
                prices = historical_data.loc[start_date:end_date, f'{coin}']
                prices.plot(ax=ax, label=coin)
        else:
            prices = historical_data.loc[start_date:end_date, f'{selected_coin}']
            prices.plot(ax=ax, label=selected_coin)

        ax.set_title(f'Price vs {currency}')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Price ({currency})')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
    except Exception as e:
        st.error(f"Error plotting price data: {str(e)}")
    
    return fig

# Updated Returns vs Volatility (Scatter with Transparency)
def plot_risk_adjusted_returns(returns, version, window=30, risk_free_rate=0.03, start_date=None, end_date=None, log_scale=False):
    """
    Plot the rolling Sharpe ratio (risk-adjusted return) for each asset over time.
    
    Parameters:
    - returns: DataFrame or dict of returns (v2 or v3 format).
    - version: 'v2' (DataFrame) or 'v3' (dict).
    - window: Rolling window size in days.
    - risk_free_rate: Annualized risk-free rate (default: 3%).
    - start_date: Start date for the plot.
    - end_date: End date for the plot.
    - log_scale: Whether to use a logarithmic y-scale.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['cyan', 'yellow', 'purple']  # Colors for BTC, ETH, XRP
    
    try:
        if version == 'v2':
            # Filter returns for the date range before calculations
            filtered_returns = returns.loc[start_date:end_date] if start_date and end_date else returns
            for i, coin in enumerate(filtered_returns.columns):
                roll_ret = filtered_returns[coin].rolling(window).mean() * 365  # Annualized return
                roll_vol = filtered_returns[coin].rolling(window).std() * np.sqrt(365)  # Annualized volatility
                sharpe_ratio = (roll_ret - risk_free_rate) / roll_vol  # Rolling Sharpe ratio
                sharpe_ratio.plot(ax=ax, color=colors[i], label=coin)
        else:
            # For version 'v3', filter each coin's returns
            for i, coin in enumerate(returns.keys()):
                coin_returns = returns[coin].loc[start_date:end_date] if start_date and end_date else returns[coin]
                roll_ret = coin_returns.rolling(window).mean() * 365
                roll_vol = coin_returns.rolling(window).std() * np.sqrt(365)
                sharpe_ratio = (roll_ret - risk_free_rate) / roll_vol
                sharpe_ratio.plot(ax=ax, color=colors[i], label=coin)
    
        # Add a horizontal line at Sharpe ratio = 0 for reference
        ax.axhline(0, color='white', linestyle='--', alpha=0.3, label='Sharpe = 0')
        
        ax.set_title(f'Rolling {window}-Day Sharpe Ratio (Risk-Adjusted Returns)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)
        if log_scale:
            ax.set_yscale('log')
    except Exception as e:
        st.error(f"Error plotting risk-adjusted returns: {str(e)}")
    
    return fig

def main():
    st.title("Crypto Portfolio Optimizer")
    
# Sidebar
    with st.sidebar:
        st.header("Portfolio Inputs")
        crypto_input = st.text_area("Enter cryptocurrency symbols (comma-separated)", "BTC, ETH, XRP")
        risk_free_rate = st.slider("Risk-free rate (%)", 0.0, 10.0, 3.0, 0.1) / 100
        version = st.selectbox("Returns Version", ['v2', 'v3'], index=1)
        custom_weights = st.text_input("Custom Weights (comma-separated, sum to 1)", "", help="e.g., 0.4,0.3,0.3")
        clean_outliers = st.checkbox("Clean Outliers", value=False)
        z_threshold = st.slider("Z-Threshold for Outliers", 1.0, 10.0, 5.0, 0.1) if clean_outliers else None
        
        # Date Filter
        st.header("Date Range")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-03-13"))
        
        st.header("Visualization Options")
        plot_options = st.multiselect("Select Plots", 
                                      ["Efficient Frontier", "Correlation Matrix", "Volatility", "Distribution", "VaR/CVaR", "Monte Carlo"],
                                      default=["Efficient Frontier"])

    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Portfolio Analysis", "Price Analysis", "About"])

    with tab1:
        coins = [coin.strip().upper() for coin in crypto_input.split(",") if coin.strip()]
        if len(coins) < 2:
            st.error("Please enter at least 2 cryptocurrencies.")
            return
        
        returns = fetch_crypto_data(coins, version, clean_outliers, z_threshold)
        
        # Compute er and cov once, outside button logic
        if version == 'v2':
            er = ct.annualize_rets(returns, 365)
            cov = returns.cov() * 365
        else:  # v3
            er = ct.annualize_rets_v3(returns, 365)
            cov = ct.cov_v3(returns).astype(float) * 365

        # Weight calculation buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Calculate GMV Weights"):
                weights = ct.get_gmv_weights(coins, version) if version == 'v3' else pd.Series(ct.gmv(cov), index=coins)
                display_weights(weights, returns, "Global Minimum Variance", version, risk_free_rate)
        with col2:
            if st.button("Calculate MSR Weights"):
                weights = pd.Series(ct.msr(risk_free_rate, er, cov), index=coins)
                display_weights(weights, returns, "Maximum Sharpe Ratio", version, risk_free_rate)
        with col3:
            if custom_weights and st.button("Use Custom Weights"):
                weights = np.array([float(w.strip()) for w in custom_weights.split(',')])
                if len(weights) == len(coins) and abs(sum(weights) - 1) < 1e-6:
                    weights = pd.Series(weights, index=coins)
                    display_weights(weights, returns, "Custom", version, risk_free_rate)
                else:
                    st.error("Weights must match number of coins and sum to 1.")

        # Portfolio Metrics
        st.subheader("Portfolio Metrics")
        method = st.selectbox("Select Portfolio", ["GMV", "MSR", "Custom"])
        if method == "GMV":
            weights = ct.get_gmv_weights(coins, version) if version == 'v3' else pd.Series(ct.gmv(cov), index=coins)
        elif method == "MSR":
            weights = pd.Series(ct.msr(risk_free_rate, er, cov), index=coins)
        else:
            weights = pd.Series([float(w.strip()) for w in custom_weights.split(',')], index=coins) if custom_weights else None
        if weights is not None:
            port_returns = (returns @ weights) if version == 'v2' else pd.concat([returns[coin] * weights[coin] for coin in returns], axis=1).sum(axis=1)
            metrics = ct.summary_stats(port_returns.to_frame('Portfolio'), riskfree_rate=risk_free_rate)
            st.dataframe(metrics)

        # for plots
        for plot in plot_options:
            st.subheader(plot)
            log_scale = st.checkbox("Log Scale", value=False, key=f"log_{plot}")
            if plot == "Efficient Frontier":
                fig = plot_efficient_frontier(er, cov, risk_free_rate, log_scale)
            elif plot == "Correlation Matrix":
                fig = plot_correlation_matrix(returns, version)
            elif plot == "Volatility":
                asset = st.selectbox("Select Asset for Volatility", ['Portfolio'] + coins, key=f"vol_{plot}")
                port_returns = returns.mean(axis=1) if version == 'v2' else pd.concat([returns[coin] for coin in coins], axis=1).mean(axis=1) if asset == 'Portfolio' else returns[asset]
                fig = plot_volatility(port_returns, asset, log_scale=log_scale)
            elif plot == "Distribution":
                asset = st.selectbox("Select Asset for Distribution", ['Portfolio'] + coins, key=f"dist_{plot}")
                port_returns = returns.mean(axis=1) if version == 'v2' else pd.concat([returns[coin] for coin in coins], axis=1).mean(axis=1) if asset == 'Portfolio' else returns[asset]
                fig = plot_returns_distribution(port_returns, asset, log_scale=log_scale)
            elif plot == "VaR/CVaR":
                asset = st.selectbox("Select Asset for VaR/CVaR", ['Portfolio'] + coins, key=f"var_{plot}")
                port_returns = returns.mean(axis=1) if version == 'v2' else pd.concat([returns[coin] for coin in coins], axis=1).mean(axis=1) if asset == 'Portfolio' else returns[asset]
                rolling = st.checkbox("Show Rolling VaR/CVaR", False, key=f"roll_{plot}")
                window = st.slider("Rolling Window (days)", 10, 100, 30, key=f"win_{plot}") if rolling else None
                fig = plot_var_cvar(port_returns, rolling, window, log_scale=log_scale)
            elif plot == "Monte Carlo":
                n_scenarios = st.slider("Number of Scenarios", 50, 500, 100, key=f"scen_{plot}")
                n_years = st.slider("Years", 1, 5, 1, key=f"yrs_{plot}")
                port_returns = returns.mean(axis=1) if version == 'v2' else pd.concat([returns[coin] for coin in coins], axis=1).mean(axis=1)
                fig = plot_monte_carlo(port_returns, n_scenarios, n_years, log_scale=log_scale)
            st.pyplot(fig, use_container_width=True)

    # New Price Analysis Tab
    with tab2:
        st.header("Price Analysis")
        currency = st.selectbox("Select Currency", ["USD", "BTC"], index=0)
        historical_data = cct.get_historical_v2(coins, currency=currency)  # Fetch once
        selected_coin = st.selectbox("Select Coin", coins + ["All"], key="price_coin_select")
        st.subheader(f"Price vs {currency}" + (f" for {selected_coin}" if selected_coin != "All" else ""))
        log_scale = st.checkbox("Log Scale", value=False, key="price_log")
        fig = plot_price_chart(historical_data, coins, currency, None if selected_coin == "All" else selected_coin, start_date, end_date, log_scale)
        st.pyplot(fig, use_container_width=True)
        
        st.subheader("Returns vs Volatility")
        log_scale = st.checkbox("Log Scale", value=False, key="rv_log")
        fig = plot_risk_adjusted_returns(returns, version, window=30, risk_free_rate=risk_free_rate, start_date=start_date, end_date=end_date, log_scale=log_scale)
        st.pyplot(fig, use_container_width=True)

    with tab3:
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