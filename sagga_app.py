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
def fetch_crypto_data(coins, version='v3', clean_outliers=False, z_threshold=5, start_date=None, end_date=None, currency='USD'):
    if version == 'v2':
        returns = cct.get_normal_returns_v2(coins, currency=currency)
        if start_date and end_date:
            returns = returns.loc[start_date:end_date]
        if clean_outliers:
            returns = ct.clean_dataframe(returns, z_threshold=z_threshold)
    elif version == 'v3':
        returns = cct.get_normal_returns_v3(coins, currency=currency)
        # if start_date and end_date:
            # returns = returns.loc[start_date:end_date]
        if clean_outliers:
            returns = ct.clean_dict(returns, z_threshold=z_threshold)
    else:
        raise ValueError("Unsupported version. Use 'v2' or 'v3'.")
    return returns

@st.cache_data(ttl=3600)
def get_gmv_weights_cached(coins, version='v3'):
    return ct.get_gmv_weights(coins, version)

@st.cache_data(ttl=3600)
def get_msr_weights_cached(risk_free_rate, er, cov):
    weights = ct.msr(risk_free_rate, er, cov)
    return weights

@st.cache_data(ttl=3600)
def compute_summary_stats(returns, riskfree_rate, version):
    if version == 'v2':
        return ct.summary_stats(returns, riskfree_rate=riskfree_rate)
    else:  # v3
        summary_data = {}
        for coin in returns:
            stats = ct.summary_stats(returns[coin].to_frame(coin), riskfree_rate=riskfree_rate)
            summary_data[coin] = stats.iloc[0]
        return pd.DataFrame(summary_data).T
    
@st.cache_data(ttl=3600)
def fetch_historical_data(coins, version='v3', currency="USD", start_date=None, end_date=None):
    if version == 'v2':
        prices = cct.get_historical_v2(coins, currency=currency)
        if start_date and end_date:
            prices = prices.loc[start_date:end_date]
    elif version == 'v3':
        prices = cct.get_historical_v3(coins, currency=currency)
        if start_date and end_date:
            prices = {coin: df.loc[start_date:end_date] for coin, df in prices.items()}
    else:
        raise ValueError("Unsupported version. Use 'v2' or 'v3'.")
    return prices

def display_weights(weights, returns, method_name, version, risk_free_rate):
    weights_df = pd.DataFrame({'Cryptocurrency': weights.index, 'Weight': weights.values * 100})
    weights_df['Weight'] = weights_df['Weight'].map("{:.2f}%".format)
    
    # display data frames
    st.dataframe(weights_df)

    # summary = compute_summary_stats(returns, risk_free_rate,)

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
    w_msr = get_msr_weights_cached(riskfree_rate, er, cov)
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
    # if asset_name == 'Portfolio':
    #     if version == 'v2':
    #         weights = weights.reindex(returns.columns).fillna(0).values
    #         data = pd.Series(ct.portfolio_return(weights, returns.values.T), index=returns.index)
    #     else:  # v3
    #         aligned_returns = pd.DataFrame({coin: returns[coin] for coin in coins}).dropna()
    #         data = aligned_returns @ weights.reindex(aligned_returns.columns).fillna(0)
    # else:
    #     data = returns[asset_name]

    data = returns
    
    sns.histplot(data.dropna(), bins=50, ax=ax, color='cyan', stat='density', label='Actual')
    mean = data.mean()
    std = data.std()

    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r-', label='Normal')
    ax.plot(x, t.pdf(x, 5, returns.mean(), returns.std()), 'g--', label='t-Student (df=5)')
    ax.axvline(mean, color='white', linestyle='--', label=f'mean: {mean:.4f}')
    ax.axvline(mean + std, color='green', linestyle='--', label='1 Std')
    ax.axvline(mean - std, color='green', linestyle='--')
    ax.axvline(mean + 2*std, color='orange', linestyle='--', label='2 Std')
    ax.axvline(mean - 2*std, color='orange', linestyle='--')
    ax.axvline(mean + norm.ppf(0.975)*std, color='purple', linestyle=':', label='Z=2')
    ax.axvline(mean - norm.ppf(0.975)*std, color='purple', linestyle=':')

    ax.set_title(f"{'Portfolio' if asset_name is None else asset_name} Returns Distribution")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

def plot_var_cvar(returns, rolling=False, window=30, days=1, log_scale=False, start_date=None, end_date=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    mean = returns.mean()
    if isinstance(returns, pd.DataFrame):
        returns = returns.mean(axis=1)  # Portfolio returns if DataFrame

    if start_date and end_date:
        returns = returns.loc[start_date:end_date]

    # Debug: Check returns statistics
    st.write(f"Returns mean: {returns.mean():.6f}, std: {returns.std():.6f}, min: {returns.min():.6f}, max: {returns.max():.6f}")
    
    returns.plot(ax=ax, color='cyan', alpha=0.5, label='Returns')
    ax.axhline(mean, color='yellow', linestyle='--', label=f'Mean Returns: {mean:.4f}')

    if rolling:
        var = ct.var_gaussian(returns, level=5, modified=True, window=window, days=days)  # Gaussian VaR with Cornish-Fisher
        cvar = ct.cvar_gaussian(returns, level=5, modified=True, window=window, days=days) # Gaussian CVaR with Cornish-Fisher
        var.plot(ax=ax, label='Rolling VaR (5%)', color='red')
        cvar.plot(ax=ax, label='Rolling CVaR (5%)', color='orange')#
        current_var = var.iloc[-1]
        current_cvar = cvar.iloc[-1]
    else:
        var = ct.var_gaussian(returns, level=5, modified=True, days=days) # Static Gaussian VaR
        cvar = ct.cvar_gaussian(returns, level=5, modified=True, days=days) # Static Gaussian CVaR
        ax.axhline(var, color='red', label=f'VaR (5%): {var:.4f}', linestyle='--')
        ax.axhline(cvar, color='orange', label=f'CVaR (5%): {cvar:.4f}', linestyle='--')
        current_var = var
        current_cvar = cvar
    ax.set_title(f'Returns vs Potential Loss (VaR/CVaR) - Current VaR: {current_var:.4f}, CVaR: {current_cvar:.4f}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

@st.cache_data(ttl=3600)
def plot_monte_carlo(returns, asset_name, n_scenarios=100, n_years=1, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    mu = ct.annualize_rets(returns, 365) # annualized returns
    sigma = ct.annualize_vol(returns, 365) # * np.sqrt(365) # annualized volatility
    sim = ct.gbm(n_years=n_years, n_scenarios=n_scenarios, mu=mu, sigma=sigma, steps_per_year=365)
    mean_trajectory = sim.mean(axis=1) # mean price at each time step

    # Plot simulations manually to control legend
    for i in range(n_scenarios):
        ax.plot(sim.index, sim.iloc[:, i], color='cyan', alpha=0.3, label=None)  # No label for individual paths
    
    # Plot only 5th and 95th percentiles with explicit labels for legend
    perc_5 = sim.quantile(0.05, axis=1)
    perc_95 = sim.quantile(0.95, axis=1)
    ax.plot(perc_5, color='red', label=f'5th Percentile (Latest: {perc_5.iloc[-1]:.2f})')
    ax.plot(perc_95, color='green', label=f'95th Percentile (Latest: {perc_95.iloc[-1]:.2f})')
    ax.plot(mean_trajectory, color='yellow', linewidth=2, label=f"Mean (Latest: {mean_trajectory.iloc[-1]:.2f})")
    ax.axhline(y=100.0, color='white', linestyle=':', alpha=0.5, label='Initial Price (100)')

    title = f"Monte Carlo Simulation - {asset_name if asset_name else 'Portfolio'}"
    ax.set_title(title)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Price')
    ax.legend()  # Only 5th and 95th percentiles will appear in the legend
    ax.grid(True, linestyle='--', alpha=0.3)
    if log_scale:
        ax.set_yscale('log')
    return fig

def plot_volatility(returns, asset_name=None, window=30, log_scale=False, start_date=None, end_date=None):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    if isinstance(returns, pd.Series):
        if start_date and end_date:
            returns = returns.loc[start_date:end_date]
        vol = returns.rolling(window).std() # * np.sqrt(365)  # Annualized volatility
        vol.plot(ax=ax, color='cyan', label=f'{asset_name} Volatility')
        current_vol = vol.iloc[-1]
    else:
        if start_date and end_date:
            returns = returns.loc[start_date:end_date]
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
def plot_price_chart(historical_data, coins, version, currency='USD', selected_coin=None, start_date=None, end_date=None, log_scale=False):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    try:
        if version == 'v2':
            if selected_coin is None:
                for coin in coins:
                    prices = historical_data.loc[start_date:end_date, f'{coin}']
                    prices.plot(ax=ax, label=coin)
            else:
                prices = historical_data.loc[start_date:end_date, f'{selected_coin}']
                prices.plot(ax=ax, label=selected_coin)

        else: #v3
            if selected_coin is None:
                for coin in coins:
                    prices = historical_data[coin].loc[start_date:end_date]
                    prices.plot(ax=ax, label=coin)
            else:
                prices = historical_data[selected_coin].loc[start_date:end_date]
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

    # expanded color palette for more assets
    colors = ['cyan', 'yellow', 'purple', 'green', 'orange', 'pink', 'blue', 'red', 'lime', 'magenta']
    
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
        clean_outliers = st.checkbox("Remove Outliers", value=True)
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
        
        with st.spinner("Fetching returns data (USD)..."):
            returns = fetch_crypto_data(coins, version, clean_outliers, z_threshold, start_date, end_date, currency='USD')
        
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
                weights = get_gmv_weights_cached(coins, version)
                display_weights(weights, returns, "Global Minimum Variance", version, risk_free_rate)
        with col2:
            if st.button("Calculate MSR Weights"):
                weights = pd.Series(get_msr_weights_cached(risk_free_rate, er, cov), index=coins)
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
            weights = get_gmv_weights_cached(coins, version)

        elif method == "MSR":
            weights = pd.Series(get_msr_weights_cached(risk_free_rate, er, cov, coins), index=coins)

        else:
            weights = pd.Series([float(w.strip()) for w in custom_weights.split(',')], index=coins) if custom_weights else None
        
        if weights is not None:
            if version == 'v2':
                # ensure weights align with returns columns
                weights = weights.reindex(returns.columns).fillna(0).values # convert to Numpy array aligned with returns
                port_returns = pd.Series(ct.portfolio_return(weights, returns.values.T), index=returns.index)  # Use weights directly with DataFrame

            else:  # v3
                aligned_returns = pd.DataFrame({coin: returns[coin] for coin in coins}).dropna()
                port_returns = aligned_returns @ weights.reindex(aligned_returns.columns).fillna(0)
            metrics = ct.summary_stats(port_returns.to_frame('Portfolio'), riskfree_rate=risk_free_rate)
            st.dataframe(metrics)

        # for plots
        for plot in plot_options:
            st.subheader(plot)
            log_scale = st.checkbox("Log Scale", value=False, key=f"log_{plot}")
            asset_key = f"asset_{plot}" # unique key for each plot's asset selector
            asset = st.selectbox("Select Asset for Plot", ['Portfolio'] + coins, key=asset_key)

            # Determine weights based on selected method
            method = st.selectbox("Select Portfolio Method", ["GMV", "MSR", "Custom"], key=f"method_{plot}")
            if method == "GMV":
                weights = get_gmv_weights_cached(coins, version)
            elif method == "MSR":
                weights = get_msr_weights_cached(risk_free_rate, er, cov, coins)
            else:  # Custom
                weights = pd.Series([float(w.strip()) for w in custom_weights.split(',')], index=coins) if custom_weights else pd.Series(np.repeat(1/len(coins), len(coins)), index=coins)
            
            # Calculate portfolio returns
            if asset == 'Portfolio':
                if version == 'v2':
                    weights = weights.reindex(returns.columns).fillna(0).values
                    port_returns = pd.Series(ct.portfolio_return(weights, returns.values.T), index=returns.index)
                else:  # v3
                    aligned_returns = pd.DataFrame({coin: returns[coin] for coin in coins}).dropna()
                    port_returns = aligned_returns @ weights.reindex(aligned_returns.columns).fillna(0)
            else:
                port_returns = returns[asset]  # Individual asset returns

            if plot == "Efficient Frontier":
                fig = plot_efficient_frontier(er, cov, risk_free_rate, log_scale)
            elif plot == "Correlation Matrix":
                fig = plot_correlation_matrix(returns, version)

            elif plot == "Volatility":
                fig = plot_volatility(port_returns, asset, log_scale=log_scale, start_date=start_date, end_date=end_date)
            elif plot == "Distribution":
                fig = plot_returns_distribution(port_returns, asset, log_scale=log_scale)
            elif plot == "VaR/CVaR":
                rolling = st.checkbox("Show Rolling VaR/CVaR", False, key=f"roll_{plot}")
                window = st.slider("Rolling Window (days)", 10, 100, 30, key=f"win_{plot}") if rolling else None
                days = st.number_input(
                    "Forecast Horizon (days)",
                    min_value=1,
                    max_value=365,
                    value=1,
                    step=1,
                    key=f"days_{plot}",
                    help="Enter the number of days in the future to estimate potential losses (VaR/CVaR)."
                )
                fig = plot_var_cvar(port_returns, rolling, window, days, log_scale=log_scale, start_date=start_date, end_date=end_date)
            elif plot == "Monte Carlo":
                n_scenarios = st.slider("Number of Scenarios", 50, 500, 100, key=f"scen_{plot}")
                n_years = st.slider("Years", 1, 5, 1, key=f"yrs_{plot}")
                fig = plot_monte_carlo(port_returns, asset, n_scenarios, n_years, log_scale=log_scale)
            st.pyplot(fig, use_container_width=True)

    # New Price Analysis Tab
    with tab2:
        st.header("Price Analysis")
        currency = st.selectbox("Select Currency", ["USD", "BTC"], index=0)
        selected_coin = st.selectbox("Select Coin", coins + ["All"], key="price_coin_select")

       # returns = fetch_crypto_data(coins, version, clean_outliers, z_threshold)

        with st.spinner(f"Fetching historical price data ({currency})..."):
            historical_data = fetch_historical_data(coins, version, currency, start_date, end_date)

        # Compute returns from historical data for Tab 2 plots
        if version == 'v2':
            if historical_data.empty:
                st.error("No historical data retrieved for the selected coins.")
                return
            tab2_r = historical_data.pct_change().dropna().replace([np.inf, -np.inf, -1], np.nan, inplace=True).fillna(0)
            st.write("Debug: tab2_r", tab2_r.head())
            r = tab2_r[selected_coin] if selected_coin != "All" else tab2_r.mean(axis=1)
        else:  # v3
            returns = {}
            for coin, coin_df in historical_data.items():
                tab2_r = coin_df.squeeze().pct_change().fillna(0)
                returns[coin] = tab2_r
            if selected_coin == "All":
                # Create DataFrame from all returns and calculate mean
                all_returns = pd.DataFrame(returns)
                r = all_returns.mean(axis=1)
            else:
                r = returns[selected_coin]
        
        st.subheader(f"Price vs {currency}" + (f" for {selected_coin}" if selected_coin != "All" else ""))
        log_scale = st.checkbox("Log Scale", value=False, key="price_log")
        fig = plot_price_chart(historical_data, coins, version, currency, None if selected_coin == "All" else selected_coin, start_date, end_date, log_scale)
        st.pyplot(fig, use_container_width=True)

        # In the "Price Analysis" tab, after "Returns Dsitrbution":
        st.subheader("Returns Distribution")
        log_scale = st.checkbox("Log Scale", value=False, key="dist_log")
        fig = plot_returns_distribution(r, selected_coin, log_scale=log_scale)
        st.pyplot(fig, use_container_width=True)
        
        st.subheader("Risk-Adjusted Returns")
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