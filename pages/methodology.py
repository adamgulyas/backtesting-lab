import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

header = st.container()
introduction = st.container()
nlp = st.container()
technical_analysis = st.container()
svm = st.container()

with header:
    st.title('Methodology')
    st.header('Here is how we made the best trading algorithm tool available on the internet')
    st.write('If you are a retail trader, you have had nightmares about this before: financial institutions have a massive leg up on you because they use algorithms to make instant calculations to determine when to buy or sell a stock. These algorithms are faster and smarter than you or I could ever be.')
    st.write('Introducing our technical analysis tool! Our streamlit web application is designed to level the playing field between you and the market makers. Below, I explain exactly how we designed this tool')
    
with nlp:
    st.header('Natural Language Processing')
    st.write('You could study techincal stock trading charts for years and still get blind-sided by news that takes the market for a ride (downward). This is where Natural Language Processing (NLP) comes in. Whenever a New York Times (NYT) article is posted, our algorithm will read the article and determine if it has a positive or negative sentiment which could impact the trend of the stock market. In the case that great or terrible news comes out, our algorithm will tell you how we think the market will move based on that data.')
    st.write('We utilized the NYT API to collect the news articles and an NLP technology called Vader to determine the sentiment of each article. After we determined the sentiment of each article on each day, we used classification to see how it impacts the stock price')
    
    df = pd.read_csv('aapl.csv')
    df
    
with technical_analysis:
    st.header('Technical Analysis')
    st.write('Humans are creatures of habit. We like to pretend that we can predict what will happen in the stock market based on what has happened in the past. As false as this is, it ends up being a self-fulfiling prophecy where if enough people follow a similar strategy, patterns in the market will persist that are somewhat reliable. In our algorithm, we rely on statistics to inform us about how the market is moving. We use the Exponentially Weighted Moving Average (EWMA), the Slow Moving Average (SMA), Moving Average Convergaence Divergence (MACD), and Bollinger Bands to determine where the stock is currently headed (up or down) and by how much. If the algorithm sees that the stock is moving upward and is trading close to the mean closing price, that would produce a signal to buy the stock. If the stock is trending up but is more than one standard deviation away from the mean (at the upper end of the Bollinger Band), that would produce a signal to sell.')
    
with svm:
    st.header('Putting It All Together')
    st.write('By combining the information that we collect through NLP and technical analysis, we can hopefully determine what direction the stock is trading, whether it is moving erratically, and what the general sentiment of the particular investment vehicle may be.')
    