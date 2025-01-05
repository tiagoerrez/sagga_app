import streamlit as st
import pandas as pd
import numpy as np

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
def fetch_crypto_data(coins):
    return cct.get_normal_returns_v3(coins)

def main():
    st.title("Crypto Portfolio Optimizer")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Portfolio Optimizer", "Efficient Frontier", "About"])
    
    with tab1:
        st.header("Portfolio Optimization")
        
        # Input for cryptocurrency symbols
        crypto_input = st.text_area(
            "Enter cryptocurrency symbols (comma-separated)",
            "BTC, ETH, XRP",
            help="Enter the symbols of cryptocurrencies you want to include in your portfolio, separated by commas."
        )
        
        # Risk-free rate input for MSR
        risk_free_rate = st.slider(
            "Risk-free rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="Annual risk-free rate used for Maximum Sharpe Ratio calculation"
        ) / 100
        
        # Convert input to list and clean it
        try:
            coins = [coin.strip().upper() for coin in crypto_input.split(",") if coin.strip()]
            
            if len(coins) < 2:
                st.error("Please enter at least 2 cryptocurrencies for portfolio optimization.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Calculate GMV Weights"):
                        with st.spinner("Calculating GMV weights..."):
                            try:
                                weights = ct.get_gmv_weights(coins, version='v3')
                                st.subheader("Global Minimum Variance Portfolio")
                                display_weights(weights)
                            except Exception as e:
                                st.error(f"Error calculating GMV weights: {str(e)}")
                
                with col2:
                    if st.button("Calculate MSR Weights"):
                        with st.spinner("Calculating Maximum Sharpe Ratio weights..."):
                            try:
                                # Get returns and covariance
                                returns = cct.get_normal_returns_v2(coins)
                                er = returns.mean() * 365  # Annualized returns
                                cov = returns.cov() * 365  # Annualized covariance
                                
                                # Calculate MSR weights
                                msr_weights = ct.msr(risk_free_rate, er, cov)
                                weights_series = pd.Series(msr_weights, index=coins)
                                
                                st.subheader("Maximum Sharpe Ratio Portfolio")
                                display_weights(weights_series)
                            except Exception as e:
                                st.error(f"Error calculating MSR weights: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
    
    with tab2:
        st.header("Efficient Frontier")
        if 'coins' in locals() and len(coins) >= 2:
            try:
                # Get returns and covariance
                returns = cct.get_normal_returns_v2(coins)
                er = returns.mean() * 365
                cov = returns.cov() * 365
                
                # Create efficient frontier plot
                fig = ct.plot_ef(
                    n_points=50,
                    er=er,
                    cov=cov,
                    show_cml=True,
                    riskfree_rate=risk_free_rate,
                    show_ew=True,
                    show_gmv=True
                )
                
                st.pyplot(fig.figure)
                
                st.write("""
                The Efficient Frontier plot shows:
                - The efficient frontier curve (blue line)
                - The Global Minimum Variance portfolio (dark blue dot)
                - The Maximum Sharpe Ratio portfolio (where the green line touches the frontier)
                - The Equal-Weight portfolio (yellow dot)
                """)
            except Exception as e:
                st.error(f"Error plotting efficient frontier: {str(e)}")
        else:
            st.info("Enter at least 2 cryptocurrencies above to view the Efficient Frontier.")
    
    with tab3:
        st.header("About Portfolio Optimization")
        st.write("""
        This dashboard implements three key portfolio optimization strategies:
        
        1. **Global Minimum Variance (GMV)**
        - Focuses on minimizing portfolio risk (volatility)
        - Does not consider expected returns
        - Useful in highly volatile markets
        
        2. **Maximum Sharpe Ratio (MSR)**
        - Optimizes the risk-adjusted return
        - Considers both returns and risk
        - Uses the risk-free rate as a reference
        
        3. **Efficient Frontier**
        - Shows all optimal portfolios for different risk levels
        - Helps visualize the risk-return tradeoff
        - Includes various reference portfolios
        """)

def display_weights(weights):
    """Helper function to display portfolio weights"""
    # Create a DataFrame for better visualization
    weights_df = pd.DataFrame({
        'Cryptocurrency': weights.index,
        'Weight': weights.values * 100  # Convert to percentage
    })
    
    # Format the weights as percentages
    weights_df['Weight'] = weights_df['Weight'].map("{:.2f}%".format)
    
    # Display the DataFrame
    st.dataframe(weights_df)
    
    # Create a pie chart
    fig = {
        'data': [{
            'labels': weights.index,
            'values': weights.values * 100,
            'type': 'pie',
            'hole': 0.4,
            'name': 'Portfolio Allocation'
        }],
        'layout': {
            'title': 'Portfolio Allocation',
            'showlegend': True
        }
    }
    
    st.plotly_chart(fig)
    
    # Add download button for the weights
    csv = weights_df.to_csv(index=False)
    st.download_button(
        label="Download Portfolio Weights",
        data=csv,
        file_name="portfolio_weights.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()