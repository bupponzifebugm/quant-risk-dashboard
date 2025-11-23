# Quant Risk Dashboard
# Author: buffon
# Description: An interactive dashboard for Risk Management & Derivatives Pricing

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Quant Risk Dashboard", layout="wide")
st.title("🛡️ Quantitative Risk & Pricing Dashboard")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("User Inputs")
ticker = st.sidebar.text_input("Enter Ticker Symbol", value='NVDA').upper()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = datetime.today()

# --- DATA LOADING ---
@st.cache_data
def get_data(ticker, start):
    try:
        data = yf.download(ticker, start=start, end=end_date)['Close']
        # Ensure it's a Series not a DataFrame (fix for new yfinance)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0] 
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = get_data(ticker, start_date)

if data is not None:
    # Calculate Returns
    returns = data.pct_change().dropna()
    current_price = data.iloc[-1]

    # --- TAB LAYOUT ---
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Risk Analysis (Fat Tails)", "🌊 Drawdown & Stress Test", "💰 Options Pricer", "🔮 Monte Carlo Simulation"])

    # ==============================================================================
    # TAB 1: DISTRIBUTION ANALYSIS (FAT TAILS)
    # ==============================================================================
    with tab1:
        st.header(f"Distribution Analysis: {ticker}")
        
        # Stats
        mu, std = returns.mean(), returns.std()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Daily Volatility", f"{std:.2%}")
        col2.metric("Annualized Volatility", f"{std * np.sqrt(252):.2%}")
        
        # VaR Calculation (95%)
        var_95 = np.percentile(returns, 5)
        price_at_risk = current_price * (1 + var_95)
        col3.metric("95% VaR (Daily)", f"{var_95:.2%}", f"Risk Level: ${current_price - price_at_risk:.2f}")

        # Plot Histogram vs Normal Distribution
        fig_hist = go.Figure()
        
        # Empirical Data (Histogram)
        fig_hist.add_trace(go.Histogram(
            x=returns, 
            histnorm='probability density', 
            name='Actual Returns',
            marker_color='#3366CC',
            opacity=0.75
        ))

        # Theoretical Data (Bell Curve)
        x_range = np.linspace(mu - 4*std, mu + 4*std, 1000)
        pdf = norm.pdf(x_range, mu, std)
        fig_hist.add_trace(go.Scatter(
            x=x_range, y=pdf, mode='lines', name='Normal Distribution',
            line=dict(color='red', width=2)
        ))

        fig_hist.update_layout(title="Empirical vs. Normal Distribution", xaxis_title="Daily Return", yaxis_title="Density")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.info("💡 **Insight:** If the Blue Bars (Actual) stick out wider than the Red Line (Normal) at the edges, the asset has 'Fat Tails'—meaning crashes happen more often than standard models predict.")


    # ==============================================================================
    # TAB 2: DRAWDOWN ANALYSIS (CALMAR)
    # ==============================================================================
    with tab2:
        st.header(f"Drawdown & Recovery Analysis: {ticker}")

        # Calculate Drawdown
        running_max = data.cummax()
        drawdown = (data - running_max) / running_max
        max_dd = drawdown.min()

        # Calculate Calmar
        days = (data.index[-1] - data.index[0]).days
        total_return = (data.iloc[-1] / data.iloc[0]) - 1
        cagr = (1 + total_return) ** (365.25 / days) - 1
        calmar = cagr / abs(max_dd)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Drawdown", f"{max_dd:.2%}")
        c2.metric("CAGR (Annual Return)", f"{cagr:.2%}")
        c3.metric("Calmar Ratio", f"{calmar:.2f}")

        # Plot Underwater Chart
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown, fill='tozeroy', name='Drawdown', line=dict(color='red')))
        fig_dd.update_layout(title="Underwater Equity Curve", yaxis_title="Drawdown %", xaxis_title="Date")
        st.plotly_chart(fig_dd, use_container_width=True)

        st.warning(f"⚠️ **Stress Test:** At the worst point in this period, investors lost **{abs(max_dd*100):.1f}%** of their capital from the peak.")


    # ==============================================================================
    # TAB 3: BLACK-SCHOLES PRICER
    # ==============================================================================
    with tab3:
        st.header("options Pricing Engine (Black-Scholes)")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            strike_price = st.number_input("Strike Price ($)", value=float(round(current_price, 0)))
            time_to_expiry = st.number_input("Time to Expiry (Years)", value=1.0)
        
        with col_b:
            risk_free_rate = st.number_input("Risk-Free Rate (%)", value=4.5) / 100
            # We use the annualized volatility we calculated earlier as the default!
            sigma_input = st.number_input("Volatility (Sigma) %", value=float(std * np.sqrt(252) * 100)) / 100

        # Black-Scholes Formula
        d1 = (np.log(current_price / strike_price) + (risk_free_rate + 0.5 * sigma_input ** 2) * time_to_expiry) / (sigma_input * np.sqrt(time_to_expiry))
        d2 = d1 - sigma_input * np.sqrt(time_to_expiry)
        
        call_price = (current_price * norm.cdf(d1)) - (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        put_price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) - (current_price * norm.cdf(-d1))

        st.divider()
        
        # Display Prices
        m1, m2 = st.columns(2)
        m1.success(f"📞 **CALL Option Price:** ${call_price:.2f}")
        m2.error(f"📉 **PUT Option Price:** ${put_price:.2f}")
        
        st.caption(f"Based on Current Price: ${current_price:.2f} | Volatility: {sigma_input:.2%}")


    # ==============================================================================
    # TAB 4: MONTE CARLO SIMULATION (THE FUTURE)
    # ==============================================================================
    with tab4:
        st.header(f"Monte Carlo Simulation: Future Paths for {ticker}")
        
        # Simulation Parameters (User Inputs)
        col_mc1, col_mc2 = st.columns(2)
        with col_mc1:
            simulations = st.slider("Number of Simulations", min_value=50, max_value=1000, value=200)
        with col_mc2:
            time_horizon = st.slider("Days into Future", min_value=30, max_value=365, value=252)

        # Run Simulation (Vectorized)
        # 1. Setup
        start_price = current_price
        daily_vol = std
        
        # 2. Generate Random Paths
        # Shape: (days, simulations)
        daily_returns_sim = np.random.normal(mu, daily_vol, (time_horizon, simulations))
        price_paths = np.zeros((time_horizon, simulations))
        
        # 3. Calculate Prices
        price_paths[0] = start_price
        for t in range(1, time_horizon):
            price_paths[t] = price_paths[t-1] * (1 + daily_returns_sim[t])
        
        # Plotting with Plotly
        fig_mc = go.Figure()
        
        # Plot the first 50 paths (to keep chart clean)
        for i in range(min(simulations, 50)):
            fig_mc.add_trace(go.Scatter(
                y=price_paths[:, i], 
                mode='lines', 
                line=dict(width=1),
                opacity=0.4,
                showlegend=False
            ))
            
        # Add Average Path
        avg_path = np.mean(price_paths, axis=1)
        fig_mc.add_trace(go.Scatter(
            y=avg_path, 
            mode='lines', 
            name='Average Path', 
            line=dict(color='red', width=3)
        ))

        fig_mc.update_layout(title=f"Projected Price Paths over {time_horizon} Days", xaxis_title="Days", yaxis_title="Price")
        st.plotly_chart(fig_mc, use_container_width=True)
        
        # Stats for the Future
        ending_values = price_paths[-1]
        expected_price = np.mean(ending_values)
        var_95_future = np.percentile(ending_values, 5)
        
        m1, m2 = st.columns(2)
        m1.metric("Expected Price (Average)", f"${expected_price:.2f}")
        m2.metric("Worst Case (95% Confidence)", f"${var_95_future:.2f}", delta_color="inverse")
        
        st.info(f"🔮 **Prediction:** Based on {simulations} simulations, there is a 95% chance the price will stay above **${var_95_future:.2f}** in {time_horizon} days.")

else:

    st.write("Please enter a valid ticker.")
