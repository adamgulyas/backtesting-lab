#!/usr/bin/env python
# coding: utf-8

# Import dependenciess
from signal import signal
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pandas.tseries.offsets import DateOffset
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from finta import TA

#Page Configuration
st.set_page_config(
    layout='wide',
    page_icon=':random:',

)

# Import data
market_data_df = pd.read_csv('data/markets_ohlc.csv', header=[0,1], index_col=0, parse_dates=True, infer_datetime_format=True)
nlp_signals_dcb = pd.read_csv('data/simple_dcb_signal.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
nlp_signals_crsh = pd.read_csv('data/simple_crsh_signal.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
nlp_signals_cvd = pd.read_csv('data/simple_cvd_signal.csv', index_col=0, parse_dates=True, infer_datetime_format=True)

# Import trained models
svm_SP500 = joblib.load('models/linear_svm_S&P 500.pkl')
svm_NASDAQ100 = joblib.load('models/linear_svm_NASDAQ 100.pkl')
svm_RUSSELL2000 = joblib.load('models/linear_svm_RUSSELL 2000.pkl')

# Declare Streamlit containers
header = st.container()
option_select = st.container()
option_select_2 = st.container()
option_select_dmac = st.container()
run = st.container()
program = st.container()
results = st.container()

#with st.sidebar:
#    st.sidebar.button('Analysis Tool')
#    st.sidebar.button('Methodology')

with header:
    st.title('Technical and Sentiment Stock Trading Algorithm')
    st.write('This web application is designed to help retail stock traders and investors determine if they should buy or sell a stock using news sentiment analysis and technical stock trading analysis. If you wish to learn more about our methodology, please visit the methodology tab')

with option_select:
    with st.form('Select Your Features'):
        col1, col2, col3 = st.columns(3)

        with col1:
            trading_strategy = st.selectbox(
            'Trading Strategy',
            ('DMAC','Linear SVM'))

        with col2:
            stock = st.selectbox(
            'Stock Market Index',
            ('S&P 500','NASDAQ 100','RUSSELL 2000'))

        with col3:
            time_period = st.selectbox(
                'Time Period',
                ('Dot-com Bubble','2008 Crash','Covid')
            )
        # st.write('You Selected:',date)

   # with option_select_2:
        col4, col5, col6 = st.columns(3)

        with col4:
            initial_capital = st.number_input(
                label='Initial Capital',
                value=10000,
                step=1000,
                min_value=1000,
                max_value=1000000000,
            )

        with col5:
            share_size = st.number_input(
                label='Share Size',
                value=500,
                step=1000,
                min_value=100,
                max_value=10000000
            )

        is_sentiment = st.checkbox('NY Times Sentiment Analysis')

        submitted = st.form_submit_button('Run My Algorithm')
with option_select_dmac:
    if trading_strategy == 'DMAC':

        col7, col8, = st.columns(2)

        with col7:
            # st.write("Fast SMA Window")
            fast_window = st.slider('Fast SMA Window',0,60,4) #min: 0, max:60, def:4

        with col8:
            # st.write('Slow SMA Window')
            slow_window = st.slider('Slow SMA Window',0,180,100) #min: 0, max:180, def:100
    if trading_strategy == 'Linear SVM':
        fast_window = 4
        slow_window = 100

#with run:
  #  if st.button('Run My Trading Algoritm'):

initial_capital = float(initial_capital)
share_size = int(share_size)
ohlc_df = market_data_df[stock]

if time_period == 'Dot-com Bubble':
    start_date = '1997-06-01'
    end_date = '2002-12-01'
elif time_period == '2008 Crash':
    start_date = '2007-06-01'
    end_date = '2012-12-01'
elif time_period == 'Covid':
    start_date = '2020-03-01'
    end_date = '2022-06-01'

ohlc_df = ohlc_df[start_date:end_date].copy()

# Helper functions

# @st.cache
def get_ohlc_data(data=pd.DataFrame, start=str, end=str):
    """
    Takes a single dimension OHLC dataframe and returns a copy of it within the
    boundaries of start and end.
    """
    return data[start:end].copy()


# Signal functions

# @st.cache
def get_under_over_signals(data=pd.DataFrame):
    """
    Create a signal based on the current day's closing price being higher or
    lower than yesterdays.

    Returns a date-indexed single column dataframe of the signal.
    """
    df = data.copy()

    df['Actual Returns'] = df['Close'].pct_change()

    df['Signal'] = 0.0
    df['Signal'] = np.where(
        (df['Actual Returns'] >= 0), 1.0, 0.0
    )

    df = df.drop(
        columns=['Close', 'Open', 'Low', 'High', 'Actual Returns']
    )
    df = df.dropna().sort_index(axis='columns')

    return df


# @st.cache
def get_fast_slow_sma(data=pd.DataFrame, fast_window=int, slow_window=int):
    """
    Create a signal based on the current day's closing price being higher or
    lower than yesterdays.

    Returns a date-indexed dataframe with SMA Fast, and SMA Slow columns
    """

    df = data.drop(columns=['Open', 'Low', 'High'])

    # Generate the fast and slow simple moving averages
    df['SMA Fast'] = (
        df['Close'].rolling(window=fast_window).mean()
    )
    df['SMA Slow'] = (
        df['Close'].rolling(window=slow_window).mean()
    )

    # Sort the index
    df = df.drop(columns='Close').dropna().sort_index(axis='columns')

    return df


# @st.cache
def get_dmac_signals(data=pd.DataFrame, fast_window=int, slow_window=int):
    """
    Create a signal based on the current day's closing price being higher or
    lower than yesterdays.

    Returns a date-indexed dataframe with SMA Fast, and SMA Slow columns
    """

    df = data.drop(columns=['Open', 'Low', 'High'])

    # Generate the fast and slow simple moving averages
    df['SMA Fast'] = (
        df['Close'].rolling(window=fast_window).mean()
    )
    df['SMA Slow'] = (
        df['Close'].rolling(window=slow_window).mean()
    )

    # Generate signals based on SMA crossovers
    df['Signal'] = 0.0
    df['Signal'][fast_window:] = np.where(
        df['SMA Fast'][fast_window:] <
        df['SMA Slow'][fast_window:], 1.0, 0.0
    )

    # Sort the index
    df = df.dropna().sort_index(axis='columns')

    return df


# @st.cache
def get_svm_signals(data=pd.DataFrame, start=str, end=str, stock=stock):
    """
    Get predicted trading signals from a Linear Support Vector Classifier
    `stock` needs to be passed in so we know which model to use
    Models are saved in the data directory as pickle files
    """

    if stock == 'S&P 500':        model = svm_SP500
    elif stock == 'NASDAQ 100':   model = svm_NASDAQ100
    elif stock == 'RUSSELL 2000': model = svm_RUSSELL2000

    # Get the appropriate feature set
    X_data = get_ohlc_data(data, start=start, end=end)
    X = get_fast_slow_sma(X_data, fast_window=fast_window, slow_window=slow_window)

    X_sc = StandardScaler().fit_transform(X)

    # Use the feature set to predict the target
    y_pred = model.predict(X_sc)

    # get the boundary datestrings of X's date-index
    X_start = X.iloc[0].name
    X_end = X.iloc[-1].name

    # Get the y_true values corresponding to the predicted set
    y_true = get_under_over_signals(X_data[X_start:X_end]).values
    y_true = np.ravel(y_true)

    df = pd.DataFrame({
        'Signal': y_pred,
        'Close': X_data[X_start:X_end]['Close']
    }, index=X.index)

    return df


# Portfolio calculation function

# @st.cache
def calculate_portfolio(data=pd.DataFrame, initial_capital=10000, share_size=500):
    """
    Calculates a running portfolio. The last row is the final result.
    Required Input: Dataframe with 'Signal' and 'Close' columns
    """

    df = data.copy()

    initial_capital = float(initial_capital)

    df['Position'] = share_size * df['Signal']
    df['Entry/Exit Position'] = df['Position'].diff()
    df['Holdings'] = df['Close'] * df['Position']
    df['Cash'] = (
        initial_capital - (df['Close'] * df['Entry/Exit Position']).cumsum()
    )
    df['Portfolio Total'] = df['Cash'] + df['Holdings']
    df['Actual Returns'] = df['Close'].pct_change()
    df['Actual Cumulative Returns'] = (
        1 + df['Actual Returns']
    ).cumprod() - 1
    df['Algorithm Returns'] = df['Actual Returns'] * df['Signal']
    df['Algorithm Cumulative Returns'] = (
        1 + df['Algorithm Returns']
    ).cumprod() - 1

    df = df.dropna().sort_index(axis='columns')

    return df



# Plotting functions

# @st.cache
def get_entries_fig(entries=pd.DataFrame):

    entries_fig = px.scatter(entries)
    entries_fig.update_traces(
        marker=dict(
            symbol='triangle-up',
            color='green',
            size=15,
            line=dict(
                    width=1,
                    color='black'
                ),
            ),
        selector=dict(mode='markers')
    )

    return entries_fig


# @st.cache
def get_exits_fig(exits=pd.DataFrame):

    exits_fig = px.scatter(exits)
    exits_fig.update_traces(
        marker=dict(
            symbol='triangle-down',
            color='red',
            size=15,
            line=dict(
                    width=1,
                    color='black'
                ),
            ),
        selector=dict(mode='markers')
    )

    return exits_fig


# @st.cache
def plot_trades(data=pd.DataFrame, stock=stock, title='Trades View'):

    df = data.copy()

    df['Entry/Exit'] = df['Signal'].diff()

    entries = df[df['Entry/Exit'] == 1.0]['Close']
    entries.rename('Buy', inplace=True)

    entries_fig = get_entries_fig(entries)

    exits = df[df['Entry/Exit'] == -1.0]['Close']
    exits.rename('Sell', inplace=True)

    exits_fig = get_exits_fig(exits)

    df = df.drop(columns=['Signal', 'Entry/Exit'])

    price_sma_fig = px.line(df)

    all_figs = go.Figure(
        data=price_sma_fig.data + entries_fig.data + exits_fig.data
    )

    all_figs.update_layout(
        # width=1200,
        # height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title
    )

    return all_figs


# @st.cache
def plot_portfolio(data=pd.DataFrame, title='Portfolio Performance'):

    df = data.copy()

    df['Entry/Exit'] = df['Signal'].diff()

    entries = df[df['Entry/Exit'] == 1.0]['Portfolio Total']
    entries.rename('Buy', inplace=True)

    entries_fig = get_entries_fig(entries)

    exits = df[df['Entry/Exit'] == -1.0]['Portfolio Total']
    exits.rename('Sell', inplace=True)

    exits_fig = get_exits_fig(exits)

    price_sma_fig = px.line(df[['Portfolio Total']])

    all_fig = go.Figure(
        data=price_sma_fig.data + entries_fig.data + exits_fig.data
    )

    all_fig.update_layout(
        # width=1200,
        # height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title)

    return all_fig


# @st.cache
def plot_returns(data=pd.DataFrame, title='Portfolio Returns'):
    """
    Plots algorithmic cumulative returns and buy & hold cumulative returns
    Input data must be a df made by `calculate_portfolio()`
    """

    df = data.copy()

    returns_fig = px.line(
        df[['Actual Cumulative Returns', 'Algorithm Cumulative Returns']]
    )

    all_fig = go.Figure(
        data=returns_fig.data
    )

    all_fig.update_layout(
        # width=1200,
        # height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title)

    return all_fig



def add_sentiment(data=pd.DataFrame):
    df = data.copy()

    if time_period == 'Dot-com Bubble': nlp_signals = nlp_signals_dcb
    elif time_period == '2008 Crash':   nlp_signals = nlp_signals_crsh
    elif time_period == 'Covid':        nlp_signals = nlp_signals_cvd

    start = df.iloc[0].name
    end = df.iloc[-1].name

    nlp_signals = nlp_signals[start:end].copy()
    df['Sentiment'] = nlp_signals

    df['Combined'] = df['Signal'] + df['Sentiment']

    df['New Signal'] = np.where(
        df['Combined'] == 2.0, 1.0, 0.0
    )

    df['Signal'] = df['New Signal'].copy()
    df = df.drop(columns=['Sentiment', 'Combined', 'New Signal'])

    return df

#------------------------------------------------------------------------------




with program:

    if trading_strategy == 'DMAC':

        signals = get_dmac_signals(ohlc_df, fast_window=fast_window, slow_window=slow_window)
        if is_sentiment: signals = add_sentiment(signals)

        portfolio = calculate_portfolio(
            signals,
            initial_capital=initial_capital,
            share_size=share_size,
        )

    elif trading_strategy == 'Linear SVM':

        signals = get_svm_signals(data=ohlc_df, start=start_date, end=end_date, stock=stock)
        if is_sentiment: signals = add_sentiment(signals)

        portfolio = calculate_portfolio(
            signals,
            initial_capital=initial_capital,
            share_size=share_size,
        )

    st.write(plot_portfolio(portfolio))
    st.write(plot_trades(signals))
    st.write(plot_returns(portfolio))


with results:
    st.header('Trading Algorithm Results')
    col6, col7, col8, col9 = st.columns(4)

    with col6:
              st.write('Cumulative Return')
              st.write("")
    with col7:
              st.write('Volatility')
              st.write("")
    with col8:
              st.write('Sharpe Ratio')
              st.write('')
    with col9:
              st.write('Sortino Ratio')
              st.write('')