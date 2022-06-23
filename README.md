# project-2: Backtesting Lab
---
### Create a platform that allows users to test a trading strategy that uses Natural Language Process, Machine Learning and Technical Indicators
---
## Description
---
* This project creates a trading app where users create a buy/sell signals of equity indices by using either one or combination of Technical Indicator, Machine Learning and Natural Language Process.
* The app was created using streamlit.  At the moment, the users are limited to one technical indicator(DMAC), support vector machine(SVM) with fixed parameters and Vader as a chosen NLP method to generate a sentiment score.
* As trading environment evolves, limited users to one technical indicator(DMAC) may not be conductive in generating the right signals.  In the future development, app will need to include more technical indicators so users to choose the appropriate indicator for the right market environment.
* Having a SVM with fixed parameters may not be optimal as each data set may need different parameters.  More research needed as to which optimal parameter's setting to achieve the desired output from the classification report.
* Vader was used  as part of NLP to create a sentiment score using articles from NY Times.  Although trading return improved when users include sentiment score.  However, more testing needed to prove the efficacy of this model.
---
## Output
* The app will produce various portfolio performance and table to enable users to assess the efficacy of their chosen strategy.  
* The following chart compares the strategy performance vs. actual performance.  This chart will help users to determine whether their chosen strategy is adding value.
https://github.com/adamgulyas/project-2/blob/main/images/port_perf.png
* The following chart shows the buy/sell signals produced from user's chosen strategy.  From this chart, users can graphically assess how well each signal contributed to the strategy's performance.

* The next following charts will demonstrate 

---
## How to use:
* Run app.py
## Python libraries:
* numpy, pandas, plotly, yfinance, finta, sklearn, nltk.sentiment.vader
## Installations:
* pip install library
## API
* https://developer.nytimes.com/ - sign up to get API key.


