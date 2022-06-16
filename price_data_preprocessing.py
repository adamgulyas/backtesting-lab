#!/usr/bin/env python
# coding: utf-8

# ### Imports

# In[ ]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from finta import TA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
import streamlit as st


# ### Declare Constants
#
# In the Streamlit app, the user will be able to choose time period as well as moving average windows, initial portfolio capital, and share size.

# In[ ]:


# The trading algorithm will be tested and evaluated over three timelines:

# dcb = Dot Com Bubble
dcb_start = '1997-06-01'
dcb_end = '2002-12-01'

# crsh = 2008 Crash
crsh_start = '2007-06-01'
crsh_end = '2012-12-01'

# cvd = COVID-19
cvd_start = '2020-03-01'
cvd_end = '2022-06-01'

short_window = 4
long_window = 100
initial_capital = 100000.0
share_size = 100
start_date = dcb_start
end_date = dcb_end


# ### Get Financial Data

# In[ ]:


# Get price data from Yahoo! Finance for S&P 500, NASDAQ 100, and RUSSELL 2000
ohlvc_df = yf.download(
    '^GSPC ^NDX ^RUT',
    start=dcb_start,
    end=cvd_end
)

# rename columns from tickers to descriptive names
ohlvc_df.rename(
    columns={
        '^NDX': 'NASDAQ 100',
        '^RUT': 'RUSSELL 2000',
        '^GSPC': 'SP 500',
    },
    inplace=True
)

ohlvc_df.drop('Adj Close', axis='columns', level=0, inplace=True)
ohlvc_df.dropna(inplace=True)
ohlvc_df.columns = ohlvc_df.columns.swaplevel(0, 1)
ohlvc_df = ohlvc_df.sort_index(axis='columns')
# ohlvc_df = ohlvc_df[start_date:end_date]


# In[ ]:


# Get the list of stocks to be used and remove duplicates
stocks = list(dict.fromkeys(ohlvc_df.columns.get_level_values(0)))
print(stocks)


# # Signal Functions

# ### `get_under_over_signals()`

# In[ ]:


def get_under_over_signals(data=pd.DataFrame):

    df = data.drop(
        columns=['Open', 'Low', 'High', 'Volume'],
        level=1,
        errors='ignore'
    )

    for stock in stocks:

        df[stock, 'Actual Returns'] = df[stock, 'Close'].pct_change()

        df[stock, 'Signal'] = 0.0
        df[stock, 'Signal'] = np.where(
            (df[stock, 'Actual Returns'] >= 0), 1, 0
        )

    return df


# ### `get_dmac_signals()`

# In[ ]:


def get_dmac_signals(data=pd.DataFrame, short_window=short_window, long_window=long_window):

    df = data.drop(
        columns=['Open', 'Low', 'High', 'Volume'],
        level=1,
        errors='ignore'
    )

    for stock in stocks:

        # Generate the fast and slow simple moving averages
        df[stock, 'SMA Fast'] = (
            df[stock, 'Close'].rolling(window=short_window).mean()
        )
        df[stock, 'SMA Slow'] = (
            df[stock, 'Close'].rolling(window=long_window).mean()
        )

        # Generate signals based on SMA crossovers
        df[stock, 'Signal'] = 0.0
        df[stock, 'Signal'][short_window:] = np.where(
            df[stock, 'SMA Fast'][short_window:] <
            df[stock, 'SMA Slow'][short_window:], 1.0, 0.0
        )

    # Sort the index
    df = df.sort_index(axis='columns')

    return df


# ### `calculate_portfolio()`

# In[ ]:


def calculate_portfolio(data=pd.DataFrame, initial_capital=initial_capital, share_size=share_size):

    df = data.copy()

    initial_capital = float(initial_capital)

    for stock in stocks:

        df[stock, 'Position'] = share_size * df[stock, 'Signal']
        df[stock, 'Entry/Exit Position'] = df[stock, 'Position'].diff()
        df[stock, 'Holdings'] = df[stock, 'Close'] * df[stock, 'Position']
        df[stock, 'Cash'] = (
            initial_capital - (df[stock, 'Close'] * df[stock, 'Entry/Exit Position']).cumsum()
        )
        df[stock, 'Portfolio Total'] = df[stock, 'Cash'] + df[stock, 'Holdings']
        df[stock, 'Actual Returns'] = df[stock, 'Close'].pct_change()
        df[stock, 'Actual Cumulative Returns'] = (
            1 + df[stock, 'Actual Returns']
        ).cumprod() - 1
        df[stock, 'Algorithm Returns'] = df[stock, 'Actual Returns'] * df[stock, 'Signal']
        df[stock, 'Algorithm Cumulative Returns'] = (
            1 + df[stock, 'Algorithm Returns']
        ).cumprod() - 1

    df = df.sort_index(axis='columns')

    return df


# ### `plot_trades()`

# In[ ]:


def plot_trades(data, stock='SP 500', title='Trades View'):

    df = data[stock].copy()

    df['Entry/Exit'] = df['Signal'].diff()

    entry_markers = df[df['Entry/Exit'] == 1.0]['Close']
    entry_markers.rename('Buy', inplace=True)

    exit_markers = df[df['Entry/Exit'] == -1.0]['Close']
    exit_markers.rename('Sell', inplace=True)

    df = df.drop(columns=['Signal', 'Entry/Exit'])

    price_sma_fig = px.line(df)

    entries_fig = px.scatter(entry_markers)
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

    exits_fig = px.scatter(exit_markers)
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

    all_fig = go.Figure(
        data=price_sma_fig.data + entries_fig.data + exits_fig.data
    )

    all_fig.update_layout(
        width=1200,
        height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title)

    return all_fig


# ### `plot_portfolio()`

# In[ ]:


def plot_portfolio(data, stock='SP 500', title='Portfolio Performance'):

    df = data[stock].copy()

    df['Entry/Exit'] = df['Signal'].diff()

    entry_markers = df[df['Entry/Exit'] == 1.0]['Portfolio Total']
    entry_markers.rename('Buy', inplace=True)

    exit_markers = df[df['Entry/Exit'] == -1.0]['Portfolio Total']
    exit_markers.rename('Sell', inplace=True)

    price_sma_fig = px.line(df[['Portfolio Total']])

    entries_fig = px.scatter(entry_markers)
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

    exits_fig = px.scatter(exit_markers)
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

    all_fig = go.Figure(
        data=price_sma_fig.data + entries_fig.data + exits_fig.data
    )

    all_fig.update_layout(
        width=1200,
        height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title)

    return all_fig


# ### `plot_returns()`

# In[ ]:


def plot_returns(data, stock='SP 500', title='Portfolio Returns'):

    df = data[stock].copy()

    df['Entry/Exit'] = df['Signal'].diff()

    # entry_markers = df[df['Entry/Exit'] == 1.0]['Portfolio Total']
    # entry_markers.rename('Buy', inplace=True)

    # exit_markers = df[df['Entry/Exit'] == -1.0]['Portfolio Total']
    # exit_markers.rename('Sell', inplace=True)

    price_sma_fig = px.line(df[[
        'Actual Cumulative Returns',
        'Algorithm Cumulative Returns'
    ]])

    # entries_fig = px.scatter(entry_markers)
    # entries_fig.update_traces(
    #     marker=dict(
    #         symbol='triangle-up',
    #         color='green',
    #         size=15,
    #         line=dict(
    #                 width=1,
    #                 color='black'
    #             ),
    #         ),
    #     selector=dict(mode='markers')
    # )

    # exits_fig = px.scatter(exit_markers)
    # exits_fig.update_traces(
    #     marker=dict(
    #         symbol='triangle-down',
    #         color='red',
    #         size=15,
    #         line=dict(
    #                 width=1,
    #                 color='black'
    #             ),
    #         ),
    #     selector=dict(mode='markers')
    # )

    all_fig = go.Figure(
        data=price_sma_fig.data
    )
    # all_fig = go.Figure(
    #     data=price_sma_fig.data + entries_fig.data + exits_fig.data
    # )

    all_fig.update_layout(
        width=1200,
        height=600,
        xaxis_title='Date',
        yaxis_title='Amount',
        title=title)

    return all_fig


# In[ ]:


# Get DMAC signals calculated from the OHLVC data set
dmac_signals_df = get_dmac_signals(ohlvc_df)
dmac_signals_df.head()

dmac_ptf_df = calculate_portfolio(dmac_signals_df)
dmac_ptf_df.head()


# In[ ]:


dmac_ptf_df = calculate_portfolio(dmac_signals_df)
dmac_ptf_df.head()


# In[ ]:

st.title('Algorithmic Trading Lab')
st.header('DMAC Strategy')
st.write(plot_trades(dmac_signals_df, 'RUSSELL 2000'))


# In[ ]:


st.write(plot_portfolio(dmac_ptf_df))


# In[ ]:


st.write(plot_returns(dmac_ptf_df, 'NASDAQ 100'))


# # SVM

# ### `get_training_dates()`

# In[ ]:


# Split any DataFrame into 75/25 train/test split
def get_training_dates(df):

    training_start = df.index.min()

    split_point = int(df.shape[0] * 0.75)
    training_end = df.iloc[split_point].name

    return training_start, training_end


# ### SVM Training and Testing

# ### `get_svm_signals()`

# In[ ]:


# Create training and testing classification reports
# train_reports = []
# test_reports = []

def get_svm_signals(data=pd.DataFrame):

    df = data.copy()

    training_start, training_end = get_training_dates(svm_features_df)

    signals_df = df[training_end:].copy()

    for stock in stocks:

        X = pd.DataFrame(index=df.index)
        X['SMA Fast'] = df[stock, 'SMA Fast'].copy()
        X['SMA Slow'] = df[stock, 'SMA Slow'].copy()

        target_df = get_under_over_signals(df)
        target = target_df[stock, 'Signal']
        y = target.dropna()

        X_train = X.loc[training_start:training_end]
        y_train = y.loc[training_start:training_end]

        X_test = X.loc[training_end:]
        y_test = y.loc[training_end:]

        scaler = StandardScaler()

        scaler = scaler.fit(X_train)
        X_train_sc = scaler.transform(X_train)
        X_test_sc = scaler.transform(X_test)

        svm = SVC()

        svm = svm.fit(X_train_sc, y_train)

        # train_signal_pred = svm.predict(X_train_sc)
        # train_reports.append(classification_report(y_train, train_signal_pred, zero_division=1))

        test_signal_pred = svm.predict(X_test_sc)
        # test_reports.append(classification_report(y_test, test_signal_pred, zero_division=1))

        signals_df[stock, 'Signal'] = test_signal_pred

    signals_df = signals_df.sort_index(axis='columns')

    return signals_df

# Print the training and testing classification reports
# [print(report) for report in train_reports]
# [print(report) for report in test_reports]


# In[ ]:


svm_features_df = get_dmac_signals(ohlvc_df).dropna().drop(
    columns=['Signal'],
    level=1,
)

svm_signals_df = get_svm_signals(svm_features_df)
svm_signals_df.head()


# In[ ]:

st.header('SVM Strategy')

st.write(plot_trades(svm_signals_df, 'NASDAQ 100'))


# In[ ]:


svm_ptf_df = calculate_portfolio(svm_signals_df)
svm_ptf_df.head()


# In[ ]:


st.write(plot_portfolio(svm_ptf_df, 'NASDAQ 100'))


# In[ ]:


st.write(plot_returns(svm_ptf_df, 'NASDAQ 100'))

