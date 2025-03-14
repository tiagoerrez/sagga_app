# import necessary libraries
import cryptocompare as cc
import pandas as pd
import numpy as np


# Get the API key
API_KEY = 'a61cf20c127429ce4b9a79fd70470ac5acf7667914fb53b2a9f8abfe5ccaea7a'

# Set the API key in the cryptocompare object
cc.cryptocompare._set_api_key_parameter(API_KEY)

print('API Key Set')


# Get number of coins returned from the API
NumberOfCoins = len(cc.get_coin_list(format=True))

# Get the list of coins from the API
def get_coin_list():
    LIST_COINS = cc.get_coin_list(format=True)
    return LIST_COINS


def get_historical(coins, currency='USD', **kwargs):
    """
    Get historical prices for a list of coins.
    DataFrame index is based on the oldest crypto-asset on the list.
    For coins that have no historical price for the oldest period, values are replaced to 0.
    """
    HISTORICAL_DF = {}

    for coin in coins:
        try:
            HISTORICAL_DF[coin] = pd.DataFrame(cc.get_historical_price_day_all(coin, currency=currency, **kwargs))
        except Exception as e:
            print(f"Error creating DataFrame: {e}")

    combined_prices_df = pd.DataFrame(columns=['Date'])

    for coin, coin_data in HISTORICAL_DF.items():
        coins_df = pd.DataFrame({'Date': coin_data['time'], coin: coin_data['close']})
        coins_df['Date'] = pd.to_datetime(coins_df['Date'], unit='s')
        coins_df.set_index('Date', inplace=True)

        combined_prices_df = pd.merge(combined_prices_df, coins_df, on='Date', how='outer')

    combined_prices_df.set_index('Date', inplace=True)
    combined_prices_df.sort_index(ascending=True, inplace=True)
    combined_prices_df.bfill(inplace=True)

    return combined_prices_df


def get_historical_v2(coins, **kwargs):
    """
    Get historical prices for a list of crypto-assets.
    Date index starts with earliest common date between the assets on the list.
    """
    
    HISTORICAL_DF = {}

    # Fetch historical data for each coin
    for coin in coins:
        HISTORICAL_DF[coin] = pd.DataFrame(cc.get_historical_price_day_all(coin, **kwargs))
        HISTORICAL_DF[coin]['Date'] = pd.to_datetime(HISTORICAL_DF[coin]['time'], unit='s')
        HISTORICAL_DF[coin].set_index('Date', inplace=True)
        HISTORICAL_DF[coin] = HISTORICAL_DF[coin][['close']].rename(columns={'close': coin})
        # Filter out rows where the 'close' price is zero
        HISTORICAL_DF[coin] = HISTORICAL_DF[coin][HISTORICAL_DF[coin][coin] != 0]

    # Find the latest common start date
    common_start_date = max(df.index.min() for df in HISTORICAL_DF.values())

    # Filter each coin's data to include only dates from the common start date
    for coin in HISTORICAL_DF:
        HISTORICAL_DF[coin] = HISTORICAL_DF[coin][HISTORICAL_DF[coin].index >= common_start_date]

    # Combine the data frames using an outer join
    combined_prices_df = pd.concat(HISTORICAL_DF.values(), axis=1, join='outer')

    combined_prices_df.sort_index(ascending=True, inplace=True)
    combined_prices_df.bfill(inplace=True)

    return combined_prices_df


def get_historical_v3_old(coins, **kwargs):

    historical_df = {}

    
    for coin in coins:
        try:
            historical_df[coin] = pd.DataFrame(cc.get_historical_price_day_all(coin, **kwargs))
        except Exception as e:
            print(f"Error processing coin {coin}: {e}")
            continue

    coins_df = {}

    for coin, coin_data in historical_df.items():
        coins_df[coin] = pd.DataFrame({'Date' : coin_data.time, coin : coin_data.close})
        coins_df[coin] = coins_df[coin][coins_df[coin][coin] != 0]
        coins_df[coin]['Date'] = pd.to_datetime(coins_df[coin]['Date'], unit='s')
        coins_df[coin].set_index('Date', inplace=True)
        coins_df[coin].sort_index(ascending=True, inplace=True)
        coins_df[coin].fillna(method='bfill', inplace=True)

    return coins_df


def get_historical_v3(coins, currency='USD', **kwargs):
    """
    Get historical prices for a list of crypto-assets.
    Gets the maximum historical data for each crypto-asset on the list.
    Returns: Dictionary of DataFrame for each cryptocurrency historical data.
    """
    
    coins_df = {}

    for coin in coins:
        try:
            # Fetch historical data for the coin
            coin_data = pd.DataFrame(cc.get_historical_price_day_all(coin, currency=currency, **kwargs))
            
            # Process the fetched data if not empty
            if not coin_data.empty and 'time' in coin_data.columns and 'close' in coin_data.columns:
                df = pd.DataFrame({'Date': coin_data['time'], coin: coin_data['close']})
                df = df[df[coin] != 0]
                df['Date'] = pd.to_datetime(df['Date'], unit='s')
                df.set_index('Date', inplace=True)
                df.sort_index(ascending=True, inplace=True)
                df.bfill(inplace=True)
                
                # Store the processed DataFrame in the dictionary
                coins_df[coin] = df[coin]
            else:
                print(f"No valid data for {coin}.")
        except Exception as e:
            print(f"Error processing coin {coin}: {e}")
            continue

    return coins_df


def get_normal_returns(coins, **kwargs):
    
    df = get_historical(coins, **kwargs)

    returns = df.pct_change()
    returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    returns.fillna(0, inplace=True)
    return returns

# Normal returns for an individual coin cleaned, excluding 0% returns

def get_returns(coins, **kwargs):
    
    df = get_historical(coins, **kwargs)

    returns = df.pct_change()
    returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    returns.dropna(inplace=True, axis=0)
    return returns


def get_normal_returns_v2(coins, **kwargs):
    
    df = get_historical_v2(coins, **kwargs)

    returns = df.pct_change()
    returns.replace([np.inf, -np.inf, -1], np.nan, inplace=True)
    returns.fillna(0, inplace=True)
    return returns


def get_normal_returns_v3(coins, period=1, **kwargs):
    df_dict = get_historical_v3(coins, **kwargs)

    returns = {}

    for coin, coin_df in df_dict.items():
        # Calculate percentage change and fill NaN values with 0
        returns_df = coin_df.squeeze().pct_change(periods=period).fillna(0)
        returns[coin] = returns_df

    return returns
