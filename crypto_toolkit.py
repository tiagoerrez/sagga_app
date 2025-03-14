import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cryptocompare_toolkit as cct
import scipy.stats as stats
from arch import arch_model
from scipy.optimize import minimize

def skewness_v3(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    This is based on the v3 returns
    """
    results = {}
    for coin, coin_data in r.items():
        results[coin] = skewness(coin_data['Price'])
    results = pd.Series(results)
    return results

def kurtosis_v3(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    This is based on the v3 returns
    """
    results = {}
    for coin, coin_data in r.items():
        results[coin] = kurtosis(coin_data['Price'])
    results = pd.Series(results)
    return results

def annualize_rets_v3(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    *This are based on the v3 returns*
    """
    ann_rets = {}
    for coin, coin_data in r.items():
        # Extract the coin name from the key by splitting on "Price"
        #coin_name = coin.split(" Price")[0]
        # ann_rets[coin_name] = annualize_rets(coin_data['Price'], periods_per_year)
        
        ann_rets[coin] = annualize_rets(coin_data, periods_per_year)
        
    ann_rets = pd.Series(ann_rets)
    return ann_rets

def annualize_vol_v3(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    *This is based in v3 returns*
    """
    ann_vol = {}
    for coin, coin_data in r.items():
        # ann_vol[coin] = annualize_vol(coin_data['Price'], periods_per_year)
        
        ann_vol[coin] = annualize_vol(coin_data, periods_per_year)
        
    ann_vol = pd.Series(ann_vol)
    return ann_vol

def cov_v3(returns_v3):
    """
    Calculates the covariance of the cryptos in coins list. This covariance calculation is
    updated based on the returns_v3 which are a dictionary of DataFrames, that is why the covariance
    can't be simmply calculated with the .cov() function.
    """
    cryptos = list(returns_v3.keys())
    cov_matrix_df = pd.DataFrame(index=cryptos, columns=cryptos)

    # Calculate the covariance between cryptocurrencies
    for crypto1 in cryptos:
        for crypto2 in cryptos:
            if crypto1 == crypto2:
                cov_matrix_df.at[crypto1, crypto2] = returns_v3[crypto1].var()
            else:
                cov = returns_v3[crypto1].cov(returns_v3[crypto2])
                cov_matrix_df.at[crypto1, crypto2] = cov
                cov_matrix_df.at[crypto2, crypto1] = cov # Covariance is symmetric
    return cov_matrix_df

def corr_v3(returns_v3):
    """
    Calculates the correlation of the cryptos in coins list. This correlation calculation is
    updated based on the returns_v3 which are a dictionary of DataFrames, that is why the correlation
    can't be simmply calculated with the .corr() function.
    """
    cryptos = list(returns_v3.keys())
    corr_matrix_df = pd.DataFrame(index=cryptos, columns=cryptos)

    # Calculate the correlation between cryptocurrencies
    for crypto1 in cryptos:
        for crypto2 in cryptos:
            if crypto1 == crypto2:
                corr_matrix_df.at[crypto1, crypto2] = 1.0 # Correlation with itself is always one
            else:
                corr = returns_v3[crypto1].corr(returns_v3[crypto2])
                corr_matrix_df.at[crypto1, crypto2] = corr
                corr_matrix_df.at[crypto2, crypto1] = corr # Correlation is symmetric
    return corr_matrix_df.astype(float)

# -------------------------------------------------------------------------------------------------------------------

def plot_returns(r, bins=100):
    for coin, coin_data in r.items():
        plt.hist(coin_data, bins=bins)
        plt.title(coin)
        plt.show()

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    # 3 should be subtracted from the final value, to give 0 for a normal distribution.
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def downside_vol(r, periods_per_year):
    """
    Computes the annualized downside volatility of a set of returns
    Only considers returns that are below zero (or a threshold like risk-free rate).
    """
    
    # Only consier negative returns
    downside_returns = np.minimum(r, 0)
    downside_deviation = downside_returns.std()
    
    # Annualize the downside volatility
    return downside_deviation * (periods_per_year ** 0.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # Convert the annual risk-free rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    
    # Apply a threshold to volatility
    min_vol_threshold = 1e-6
    ann_vol = np.where(ann_vol < min_vol_threshold, np.nan, ann_vol)
    
    sharpe_ratio = ann_ex_ret / ann_vol
    
    return sharpe_ratio

def sortino_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized Sortino Ratio of a set of returns
    """
    
    # Convert the annual risk-free rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    
    # Annualize excess returns
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    
    # Annualize downside volatility
    ann_down_vol = downside_vol(excess_ret, periods_per_year)
    
    # Apply a threshold to downside volatility
    # This avoids dividing by zero in case of azero or negligible downside volatility
    min_vol_threshold = 1e-6
    ann_down_vol = np.where(ann_down_vol < min_vol_threshold, np.nan, ann_down_vol)
    
    sortino_ratio = ann_ex_ret / ann_down_vol
    
    return sortino_ratio
    
def max_drawdown(returns):
    """
    Calculate the maximum drawdown of a series of returns.
    """
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return dd.min()

def calmar_ratio(returns, periods_per_year):
        """
        Calculate the Calmar ratio of a series of returns
        """
        ann_return = annualize_rets(returns, periods_per_year)
        max_dd = max_drawdown(returns)
        return ann_return / abs(max_dd)
    
def upside_potential_ratio(returns, riskfree_rate, periods_per_year):
    """
    Calculate the Upside Potential Ratio of a series of returns.
    """
    excess_returns = returns - (riskfree_rate / periods_per_year)
    upside_potential = np.sqrt(np.mean(excess_returns[excess_returns > 0] ** 2))
    downside_deviation = np.sqrt(np.mean(excess_returns[excess_returns < 0] ** 2))
    return upside_potential / downside_deviation

def omega_ratio(returns, riskfree_rate, periods_per_year):
    """
    Calculate the Omega ratio of a series of returns.
    """
    excess_returns = returns - (riskfree_rate / periods_per_year)
    return np.exp(np.mean(excess_returns[excess_returns > 0])) / np.exp(np.mean(-excess_returns[excess_returns < 0]))

import scipy.stats

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    returns a DataFrame with columns for
    the wealth index, 
    the previous peaks, and 
    the percentage drawdown
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({'Wealth': wealth_index,
                         'Previous Peak': previous_peaks,
                         'Drawdown': drawdowns})

def semideviation(r):
    """
    Returns the semidiviation aka negative semidiviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    if isinstance(r, pd.Series):
        is_negative = r < 0
        return r[is_negative].std(ddof=0)
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(semideviation)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def var_historic(r, level=5, days=1, window=None):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level, days=days)
    
    elif isinstance(r, pd.Series):
        
        if window is None:
            return -np.percentile(r, level) * np.sqrt(days)
        else:
            rolling_window = r.rolling(window)
            percentiles = rolling_window.apply(lambda x: np.percentile(x, level), raw=True)
            return -(percentiles * np.sqrt(days)).dropna()
        
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        

def cvar_historic(r, level=5, days=1, window=None):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        
        if window is None:
            var_value = var_historic(r, level=level, days=1)
            is_beyond = r <= -var_value
            return -r[is_beyond].mean() * np.sqrt(days)
        
        else:
            rolling_window = r.rolling(window)
            cvar = rolling_window.apply(lambda x: x[x <= np.percentile(x, level)].mean(), raw=True)
            return -(cvar * np.sqrt(days)).dropna()
    
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level, days=days, window=window)
    
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
    
from scipy.stats import norm

def rolling_var(x, level, modified, days):
    z = norm.ppf(level / 100)
    if modified:
        s = stats.skew(x)
        k = stats.kurtosis(x, fisher=False)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3 * z) * (k - 3) / 24 -
             (2 * z**3 - 5 * z) * (s**2) / 36)
    return -(x.mean() + z * x.std(ddof=0)) * np.sqrt(days)


def rolling_cvar(x, level, days, modified):
    z = norm.ppf(level / 100)
    if modified:
        s = stats.skew(x)
        k = stats.kurtosis(x, fisher=False)
        z = (z +
             (z**2 - 1) * s / 6 +
             (z**3 - 3 * z) * (k - 3) / 24 -
             (2 * z**3 - 5 * z) * (s**2) / 36)
    # cvar = -(x.mean() + (norm.pdf(z) / (1 - level / 100)) * x.std(ddof=0))
    cvar = (x.mean() + (norm.pdf(z) / (level / 100)) * x.std(ddof=0))
    return cvar * np.sqrt(days)


def var_gaussian(r, level=5, modified=False, days=1, window=None):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    
    if window is None:
        # compute the Z score assuming it was Gaussian
        z = norm.ppf(level/100)
        if modified:
            # modify the Z score based on observed skewness and kurtosis
            s = skewness(r)
            k = kurtosis(r)
            z = (z +
                 (z**2 - 1)*s/6 +
                 (z**3 -3*z)*(k-3)/24 -
                 (2*z**3 - 5*z)*(s**2)/36
                )
        return -(r.mean() + z*r.std(ddof=0)) * np.sqrt(days)
    
    else:
        return r.rolling(window).apply(lambda x: rolling_var(x, level, days, modified), raw=False).dropna()
        


def cvar_gaussian(r, level=5, modified=False, days=1, window=None):
    """
    Returns the Parametric Gaussian CVaR of a Series or DataFrame.
    If "modified" is True, then the modified CVaR is returned,
    using the Cornish-Fisher modification.
    
    Parameters:
    r : Series or DataFrame
        The return data.
    level : float
        The confidence level for the VaR (e.g., 5 for 5% VaR).
    modified : bool
        Whether to use the Cornish-Fisher modification for skewness and kurtosis.
        
    Returns:
    float
        The Conditional Value at Risk (CVaR).
    """
    if window is None:
        # Compute the Z score assuming it was Gaussian
        z = stats.norm.ppf(level / 100)
        if modified:
            # Modify the Z score based on observed skewness and kurtosis
            s = stats.skew(r)
            k = stats.kurtosis(r, fisher=False)
            z = (z +
                 (z**2 - 1) * s / 6 +
                 (z**3 - 3 * z) * (k - 3) / 24 -
                 (2 * z**3 - 5 * z) * (s**2) / 36
                )
        
        # Calculate CVaR
        # For Gaussian distribution, CVaR can be derived as follows:
        # cvar = -(r.mean() + (stats.norm.pdf(z) / (1 - level / 100)) * r.std(ddof=0))
        cvar = -(r.mean() + r.std(ddof=0) * np.sqrt(days) * stats.norm.pdf(z) / ( (1-level) / 100))
        return cvar
    
    else:
        return r.rolling(window).apply(lambda x: rolling_cvar(x, level, days, modified), raw=False).dropna()


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    return (weights.T @ covmat @ weights)**0.5

def minimize_vol(target_return, er, cov):
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def get_gmv_weights(coins, version='v3'):
    '''
    Returns the weights of the portfolio that gives the optomized GMV
    given a set of coins.
    version: refers to which version of returns is, either v1, v2 or v3. This
    affects how returns are calculated and how the covariance matrix is calculated.
    '''
    
    if version == 'v2':
        e_v2 = cct.get_normal_returns_v2(coins)
        cov_v2 = e_v2.cov()
        gmv_v2 = gmv(cov_v2)
        
        gmv_results = list(zip(coins, gmv_v2))
        gmv_results = pd.Series(dict(gmv_results)).sort_values(ascending=False).round(5)
        
        return gmv_results


    elif version == 'v4':
        e_v4 = cct.get_normal_returns_v4(coins)
        cov_v4 = e_v4.cov()
        gmv_v4 = gmv(cov_v4)
        
        gmv_results = list(zip(coins, gmv_v4))
        gmv_results = pd.Series(dict(gmv_results)).sort_values(ascending=False).round(5)
        
        return gmv_results        
        

    elif version == 'v3':
        e_v3 = cct.get_normal_returns_v3(coins)
        e_v3_cleaned = clean_dict(e_v3)
        covv3 = cov_v3(e_v3_cleaned)
        gmv_v3 = gmv(covv3)
        
        gmv_results = list(zip(coins, gmv_v3))
        gmv_results = pd.Series(dict(gmv_results)).sort_values(ascending=False).round(5)
        
        return gmv_results

    else:
        raise ValueError("Unsupported version. Please use 'v2' or 'v3.'")


def optimal_weights(n_points, er, cov):
    """
    Returns a list of weights that represent a grid of n_points on the efficient frontier
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the multi-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, label="Efficient Frontier")
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax

def calculate_volatility(returns, method, lookback):
    """
    Calculate volatility using specified method.
    """
    
    if method == 'ewma':
        upside_vol = returns[returns > 0].ewm(span=lookback).std().iloc[-1]
        downside_vol = returns[returns < 0].ewm(span=lookback).std().iloc[-1]
        
    elif method == 'garch':
        returns_scaled = returns * 1000
        am = arch_model(returns_scaled, mean='zero', vol='GARCH', p=1, q=1, dist='t', rescale=False)
        res = am.fit(disp='off')
        volatility = res.conditional_volatility.iloc[-1] / 1000
        upside_vol = downside_vol = volatility
        
    elif method == 'hist':
        upside_vol = returns[returns > 0].std()
        downside_vol = returns[returns < 0].std()
        
    else:
        raise ValueError("Invalid vol_method specified. Choose 'ewma', 'garch', or 'hist'.")
    
    return np.nan_to_num(upside_vol), np.nan_to_num(downside_vol)


def adjust_multiplier(m, upside_vol, downside_vol, vol_scaling, method, m_floor, m_cap):
    """
    Adjust multiplier based on volatility.
    """
    
    if method == 'sigmoid':
        vol_diff = upside_vol - downside_vol
        dynamic_m = m * (1 + vol_scaling * (1 / (1 + np.exp(-vol_diff)) - 0.5))
        
    elif method == 'piecewise':
        dynamic_m = m * (1 + vol_scaling * (upside_vol - downside_vol))
        
    elif method == 'tanh':
        vol_diff = upside_vol - downside_vol
        dynamic_m = m * (1 + vol_scaling * np.tanh(vol_diff))
        
    else:
        raise ValueError("Invalid m_adjust_func specified. Choose 'sigmoid', 'piecewise', or 'tanh'.")
    
    return np.clip(dynamic_m, m_floor, m_cap)


def run_cppi(risky_r, safe_r=None, m=3, start=1000,
             floor=0.75,riskfree_rate=0.03, drawdown=None,
             dynamic_m=True, periods_per_year=365, vol_lookback=30,
             vol_scaling=1.5, m_floor=1, m_cap=6, vol_method='ewma', m_adjust_func='sigmoid'):
    """
    Run a backtest of the CPPI strategy with dynamic multiplier based on upside and downside volatility.
    
    Parameters:
    - risky_r: pd.Series or pd.DataFrame of risky asset returns
    - safe_r: pd.Series or pd.DataFrame of safe asset returns (default: constant risk-free rate)
    - m: base multiplier
    - start: starting portfolio value
    - floor: floor percentage of starting value
    - riskfree_rate: annual risk-free rate
    - drawdown: maximum acceptable drawdown (e.g., 0.2 for 20%)
    - dynamic_m: boolean to enable dynamic adjustment of m
    - periods_per_year: number of trading periods per year (e.g., 252 for daily data)
    - vol_lookback: lookback period for volatility calculation
    - vol_scaling: scaling factor for adjusting m based on volatility differences
    - m_floor: minimum value for dynamic m
    - m_cap: maximum value for dynamic m
    
    Returns:
    - backtest_result_df: pd.DataFrame containing portfolio metrics over time
    """
    # Ensure returns are in DataFrame format
    if isinstance(risky_r, pd.Series): 
        risky_r = risky_r.to_frame("RiskyAsset")
    if safe_r is None:
        safe_r = pd.DataFrame(riskfree_rate / periods_per_year, index=risky_r.index, columns=["SafeAsset"])
    else:
        if isinstance(safe_r, pd.Series):
            safe_r = safe_r.to_frame("SafeAsset")
    
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = account_value
    
    # Initialize DataFrames to store historical values
    history = pd.DataFrame(index = dates,
                           columns=['Wealth', 'RiskyAllocation', 'Multiplier',
                                    'Cushion', 'Floor', 'Peak'],
                           dtype='float64')
    
    # Ensure vol_lookback is an integer and a valid lookback period
    vol_lookback = int(max(5, vol_lookback))  # Set a minimum lookback period of 5
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = max(peak, account_value)
            floor_value = peak * (1 - drawdown)

        cushion = (account_value - floor_value) / account_value
        
        if dynamic_m and step >= vol_lookback:
            upside_vol, downside_vol = calculate_volatility(risky_r.iloc[step-vol_lookback:step], vol_method, vol_lookback)
            dynamic_m_value = adjust_multiplier(m, upside_vol, downside_vol, vol_scaling, m_adjust_func, m_floor, m_cap)
        else:
            dynamic_m_value = m

        risky_w = np.clip(dynamic_m_value * cushion, 0, 1)
        safe_w = 1 - risky_w

        # Convert array-like values to scalars using `.iloc[step].values[0]`
        risky_return = float(risky_r.iloc[step].values[0]) if isinstance(risky_r.iloc[step].values, np.ndarray) else float(risky_r.iloc[step])
        safe_return = float(safe_r.iloc[step].values[0]) if isinstance(safe_r.iloc[step].values, np.ndarray) else float(safe_r.iloc[step])
    
        # Update account value
        account_value = account_value * (risky_w * (1 + risky_return) + 
                                         safe_w * (1 + safe_return))
    
        # Record history, ensuring scalar values are assigned
        history.loc[dates[step]] = [
            float(account_value),           # Ensure scalar
            float(risky_w),                 # Ensure scalar
            float(dynamic_m_value),         # Ensure scalar
            float(cushion),                 # Ensure scalar
            float(floor_value),             # Ensure scalar
            float(peak)                     # Ensure scalar
        ]

    # Calculate performance metrics
    wealth_returns = history['Wealth'].pct_change().dropna()
    history['SharpeRatio'] = sharpe_ratio(wealth_returns, riskfree_rate, periods_per_year)
    history['SortinoRatio'] = sortino_ratio(wealth_returns, riskfree_rate, periods_per_year)
    history['MaxDrawdown'] = max_drawdown(wealth_returns)
    history['CalmarRatio'] = calmar_ratio(wealth_returns, periods_per_year)
    history['UpsidePotentialRatio'] = upside_potential_ratio(wealth_returns, riskfree_rate, periods_per_year)
    history['OmegaRatio'] = omega_ratio(wealth_returns, riskfree_rate, periods_per_year)
    
    history['RiskyWealth'] = start * (1 + risky_r).cumprod()
    history['SafeWealth'] = start * (1 + safe_r).cumprod()
    history['risky_r'] = risky_r.values

    return history


def gbm(n_years = 5, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, steps_per_year=12, y_max=100, drawdown=0.3):
    """
    Plot the results of a Monte Carlo Simulation of CPPI
    """
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # run the "back"-test
    btr = run_cppi(risky_r=pd.DataFrame(risky_r),riskfree_rate=riskfree_rate,m=m, start=start, floor=floor, drawdown=drawdown)
    wealth = btr["wealth"]

    # calculate terminal wealth stats
    y_max=wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios

    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0

    # Plot!
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3,2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color="indianred")
    wealth_ax.axhline(y=start, ls=":", color="black")
    wealth_ax.axhline(y=start*floor, ls="--", color="red")
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=":", color="black")
    hist_ax.axhline(y=tw_mean, ls=":", color="blue")
    hist_ax.axhline(y=tw_median, ls=":", color="purple")
    hist_ax.annotate(f"Mean: ${int(tw_mean)}", xy=(.7, .9),xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f"Median: ${int(tw_median)}", xy=(.7, .85),xycoords='axes fraction', fontsize=24)
    if (floor > 0.01):
        hist_ax.axhline(y=start*floor, ls="--", color="red", linewidth=3)
        hist_ax.annotate(f"Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}", xy=(.7, .7), xycoords='axes fraction', fontsize=24)


def clean_dict(data_dict, z_threshold=5):
    """
    Cleana dictionary of time series data for multiple coins.
    
    Parameters:
    data_dict (dict): Dictionary where keys are coin names and values are lists of returns.
    z_threshold (float): Z-score threshold to identify outliers.
    
    Returns:
    dict: Cleaned dictionary of coin returns.
    """
    cleaned_dict = {}
    
    for coin, returns in data_dict.items():
        series = pd.Series(returns)
        
        # Calculate Z-scores
        z_scores = (series - series.mean()) / series.std()
        
        # Interpolate to replace outliers
        series_interpolated = series.copy()
        series_interpolated[np.abs(z_scores) >= z_threshold] = np.nan
        series_interpolated.interpolate(method='linear', inplace=True)
        
        # update the cleaned_dict with the interpolated series
        cleaned_dict[coin] = series_interpolated
    
    return cleaned_dict

def clean_dataframe(data, **kwargs):
    """
    Clean a DataFrame of time series data for multiple coins or a single Series.
    
    Parameters:
    data (pd.DataFrame or pd.Series): DataFrame where each column is the returns of a coin, or a Series.
    z_threshold (float): Z-score threshold to identify outliers.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame of coin returns if input is a DataFrame,
                  Cleaned Series if input is a Series.
                  
    """
    
    z_threshold = kwargs.get('z_threshold', 5)
    
    if isinstance(data, pd.DataFrame):
        cleaned_data = data.copy()

        for column in cleaned_data.columns:
            series = cleaned_data[column]

            # Calculate Z-scores
            z_scores = (series - series.mean()) / series.std()

            # Interpolate to replace outliers
            series_interpolated = series.copy()
            series_interpolated[np.abs(z_scores) >= z_threshold] = np.nan
            series_interpolated.interpolate(method='linear', inplace=True)

            # Update the cleaned DataFrame with the interpolated series
            cleaned_data[column] = series_interpolated

        return cleaned_data

    elif isinstance(data, pd.Series):
        series = data.copy()

        # Calculate Z-scores
        z_scores = (series - series.mean()) / series.std()

        # Interpolate to replace outliers
        series_interpolated = series.copy()
        series_interpolated[np.abs(z_scores) >= z_threshold] = np.nan
        series_interpolated.interpolate(method='linear', inplace=True)

        return series_interpolated

    else:
        raise ValueError("Input must be a DataFrame or a Series.")

def plot_cppi(cppi_data):
    """
    Plots CPPI results for Wealth, Risky Wealth, floor and peak
    
    Parameters: ccpi_dict needs to be a dictionary containing the above DataFrames or Series
    """
    
    # handle for dictionaries
    if isinstance(cppi_data, dict):
        # Extract data from dict
        coin_name = cppi_data['Wealth'].columns[0]
        wealth_col = cppi_data['Wealth'][coin_name]
        risky_wealth_col = cppi_data['RiskyWealth'][coin_name]
        floor_col = cppi_data['Floor'][coin_name]
        peak_col = cppi_data['Peak'][coin_name]
    
    elif isinstance(cppi_data, pd.DataFrame):
        # Dynamic column names
        coin_name = cppi_data.columns[-1]
        wealth_col = cppi_data['Wealth']
        risky_wealth_col = cppi_data['RiskyWealth']
        floor_col = cppi_data['Floor']
        peak_col = cppi_data['Peak']
    else:
        raise ValueError("Input must be a dictionary or DataFrame")
    
    ax = wealth_col.plot(legend=True, figsize=(10, 5), label=f'{coin_name} Wealth')
    risky_wealth_col.plot(ax=ax, style='--', legend=True, label=f'{coin_name} Risky Wealth')
    floor_col.plot(ax=ax, style=':', legend=True, label=f'{coin_name} floor')
    peak_col.plot(ax=ax, style=':', legend=True, label=f'{coin_name} peak')
    
    plt.title('CPPI Strategy Performance')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()
    
    
def summary_stats(r, riskfree_rate=0.03):
    """
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_r = r.aggregate(annualize_rets, periods_per_year=365)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=365)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=365)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
    

def cppi_sr(coin_list, m=4, floor=.75, drawdown=.25, **kwargs):

    """
    Returns a DataFrame or Pandas Series of Sharpe Ratios for a given list of coins and a given CPPI strategy.
    Results are sorted by highest to lowest Sharpe Ratios.
    """
    
    # Create the set of returns
    try:
        returns = cct.get_normal_returns_v3(coin_list, **kwargs)
    except Exception as e:
        print(f"Error fetching returns for coin: {e}")
        return None
    
    # clean the returns for outliers with a given z-threshold
    clean_returns = clean_dict(returns, **kwargs)
    
# Create an empty dict to store the Sharpe Ratios
    sharpe_ratios = {}
    cppi_results = {}
    
    # For loop to create the CPPI for all of the coins in the list
    for coin, coin_data in clean_returns.items():
        
        try:
            cppi_result = run_cppi(coin_data, m=m, floor=floor, drawdown=drawdown, **kwargs)
            cppi_result['wealth'] = cppi_result['wealth'].astype(float)
            
            cppi_results[coin] = cppi_result  # Store CPPI results
            
            # Calculate wealth returns, which are the CPPI strategy returns
            wealth_returns = cppi_result['wealth'].pct_change().fillna(0)
            
            # Calculate the Sharpe Ratio for each coin's CPPI strategy results
            sharpe_ratios[coin] = sharpe_ratio(wealth_returns, periods_per_year=365, riskfree_rate=0.03)
            
        except Exception as e:
            print(f"Error processing coin {coin}: {e}")
            continue
    
    # Convert the Sharpe Ratios to a Pandas Series and sort them
    sharpe_ratio_series = pd.Series(sharpe_ratios).sort_values(ascending=False)
    
    return sharpe_ratio_series, cppi_results
    
    


