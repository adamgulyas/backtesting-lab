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
from itertools import groupby
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
trades = st.container()

with header:
    st.title('Backtesting Lab')
    st.write('This application is designed to help retail stock traders and investors determine if they should buy or sell a stock using news sentiment analysis and technical stock trading analysis. If you wish to learn more about our methodology, please visit the methodology tab')

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
                value=10,
                step=1000,
                min_value=1,
                max_value=1000
            )

        is_sentiment = st.checkbox('NY Times Sentiment Analysis')

        submitted = st.form_submit_button('Run Algorithm')

with option_select_dmac:

    if trading_strategy == 'DMAC':

        col7, col8, = st.columns(2)

        with col7:
            fast_window = st.slider('Fast SMA Window',0,60,4) #min: 0, max:60, def:4

        with col8:
            slow_window = st.slider('Slow SMA Window',0,180,100) #min: 0, max:180, def:100

    if trading_strategy == 'Linear SVM':

        fast_window = 4
        slow_window = 100

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
    df['Entry/Exit'] = df['Signal'].diff()
    df['Portfolio Holdings'] = (
        df["Close"] * df["Entry/Exit Position"].cumsum()
    )
    df['Cash'] = (
        initial_capital - (df['Close'] * df['Entry/Exit Position']).cumsum()
    )
    df['Portfolio Total'] = df['Cash'] + df['Portfolio Holdings']
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



def evaluate_portfolio(data=pd.DataFrame):

    df = data.copy()

    # Create a list for the column name
    columns = ["Backtest Results"]

    # Create a list holding the names of the new evaluation metrics
    metrics = [
        "Annualized Return",
        "Cumulative Returns",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Best Winning Streak",
        "Worst Losing Streak"]

    # Initialize the DataFrame with index set to the evaluation metrics and the column
    ptf_eval_df = pd.DataFrame(index=metrics, columns=columns)

    # Calculate annualized return
    ptf_eval_df.loc["Annualized Return"] = df.mean() * 252

    # Calculate cumulative return

    ptf_eval_df.loc["Cumulative Returns"] = ((1 + df.cumsum()) - 1)[-1]


    # Calculate annual volatility
    ptf_eval_df.loc["Annual Volatility"] = df.std() * np.sqrt(252)

    # Calculate Sharpe ratio
    ptf_eval_df.loc["Sharpe Ratio"] = \
        df.mean() * 252 / (df.std() * np.sqrt(252))


    # Create a DataFrame that contains the Portfolio Daily Returns column
    sortino_ratio_df = df.copy()

    # The Sortino ratio is reached by dividing the annualized return value
    # by the downside standard deviation value
    sortino_ratio = df.mean() * 252 / (df[df < 0].std() * np.sqrt(252))

    # Add the Sortino ratio to the evaluation DataFrame
    ptf_eval_df.loc["Sortino Ratio"] = sortino_ratio

    # Best and Worst streak
    from itertools import groupby

    # Best winning streak
    L = df.copy()
    L[L > 0] = 1
    L[L < 0] = float("NaN")
    longest = max((list(g) for _, g in groupby(L)), key=len)
    ptf_eval_df.loc["Best Winning Streak"] = len(longest)

    # Worst losing streak
    L = df.copy()
    L[L < 0] = -1
    L[L > 0] = float("NaN")
    longest = max((list(g) for _, g in groupby(L)), key=len)
    ptf_eval_df.loc["Worst Losing Streak"] = len(longest)

    return ptf_eval_df



def evaluate_trades(data=pd.DataFrame):

    df = data.copy()

    trade_eval_df = pd.DataFrame(
        columns=[
            "Stock",
            "Entry Date",
            "Exit Date",
            "Shares",
            "Entry Share Price",
            "Exit Share Price",
            "Entry Portfolio Holding",
            "Exit Portfolio Holding",
            "Profit/Loss"]
    )

    for index, row in df.iterrows():

        if row['Entry/Exit'] == 1:
            entry_date = index
            entry_portfolio_holding = row['Portfolio Holdings']
            share_size = row['Entry/Exit Position']
            entry_share_price = row['Close']

        elif row['Entry/Exit'] == -1:
            exit_date = index
            exit_portfolio_holding = abs(row['Close'] * row['Entry/Exit Position'])
            exit_share_price = row['Close']
            profit_loss = exit_portfolio_holding - entry_portfolio_holding
            trade_eval_df = trade_eval_df.append(
                {
                    'Stock': stock,
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Shares': share_size,
                    'Entry Share Price': entry_share_price,
                    'Exit Share Price': exit_share_price,
                    'Entry Portfolio Holding': entry_portfolio_holding,
                    'Exit Portfolio Holding': exit_portfolio_holding,
                    'Profit/Loss': profit_loss
                },
                ignore_index=True)


    return trade_eval_df



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
        width=1400,
        height=700,
        xaxis_title='Date',
        yaxis_title='Amount',
        # title=title
    )

    return all_figs


# @st.cache
def plot_portfolio(data=pd.DataFrame, title='Portfolio Performance'):

    df = data.copy()

    df['Entry/Exit'] = df['Signal'].diff()

    entries = df[df['Entry/Exit'] == 1.0]['Algorithm Cumulative Returns']
    entries.rename('Buy', inplace=True)

    entries_fig = get_entries_fig(entries)

    exits = df[df['Entry/Exit'] == -1.0]['Algorithm Cumulative Returns']
    exits.rename('Sell', inplace=True)

    exits_fig = get_exits_fig(exits)

    df.rename(columns={'Algorithm Cumulative Returns': 'Algorithm'}, inplace=True)
    price_sma_fig = px.line(df[['Algorithm']])

    all_fig = go.Figure(
        data=price_sma_fig.data + entries_fig.data + exits_fig.data
    )

    all_fig.update_layout(
        width=1400,
        height=700,
        xaxis_title='Date',
        yaxis_title='Amount',
        # title=title
    )

    return all_fig


# @st.cache
def plot_returns(data=pd.DataFrame, title='Portfolio Returns'):
    """
    Plots algorithmic cumulative returns and buy & hold cumulative returns
    Input data must be a df made by `calculate_portfolio()`
    """

    df = data[['Actual Cumulative Returns', 'Algorithm Cumulative Returns']].copy()
    df.columns = ['Actual', 'Algorithm']

    returns_fig = px.line(df)

    all_fig = go.Figure(
        data=returns_fig.data
    )

    all_fig.update_layout(
        width=1400,
        height=700,
        xaxis_title='Date',
        yaxis_title='Amount',
        # title=title
    )

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

    st.write('#')
    st.write('#')
    st.header('Portfolio Returns')
    st.write(plot_returns(portfolio))
    st.header('Portfolio Performace')
    st.write(plot_portfolio(portfolio))
    st.header('Entry/Exit Signals')
    st.write(plot_trades(signals))


with results:
    st.header('Portfolio Evaluation')
    col6, col7, col8, col9 = st.columns(4)

    ptf_eval = evaluate_portfolio(portfolio['Algorithm Returns'])

    with col6:
        st.write('Cumulative Return')
        cum_returns = portfolio['Algorithm Cumulative Returns'][-1]
        cum_returns_price = cum_returns * initial_capital
        st.subheader(f'${round(cum_returns_price, 2)}')

    with col7:
        st.write('Annualized Return')
        ann_returns = ptf_eval.loc['Annualized Return', 'Backtest Results']
        st.subheader(f'${round(ann_returns * initial_capital, 2)}')

    with col8:
        st.write('Portfolio Total')
        st.subheader(f'${round(initial_capital + cum_returns_price, 2)}')


    with col9:
        st.write('Best Winning Streak')
        best_streak = ptf_eval.loc['Best Winning Streak', 'Backtest Results']
        st.subheader(f'${int(best_streak)}')

    st.write('#')

    col10, col11, col12, col13 = st.columns(4)

    with col10:
        st.write('Market Return')
        market_returns = portfolio['Actual Cumulative Returns'][-1]
        market_returns_price = market_returns * initial_capital
        st.subheader(f'${round(market_returns_price, 2)}')

    with col11:
        st.write('Annual Volatility')
        ann_vol = ptf_eval.loc['Annual Volatility', 'Backtest Results']
        st.subheader(round(ann_vol, 3))

    with col12:
        st.write('Sharpe Ratio')
        sharpe = ptf_eval.loc['Sharpe Ratio', 'Backtest Results']
        st.subheader(round(sharpe, 3))

    with col13:
        st.write('Sortino Ratio')
        sortino = ptf_eval.loc['Sortino Ratio', 'Backtest Results']
        st.subheader(round(sortino, 3))

with trades:
    st.write('#')
    st.header('Trades Evaluation')
    trades = evaluate_trades(portfolio)
    st.table(trades)