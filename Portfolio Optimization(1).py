#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install yfinance
#pip install PyPortfolioOpt
#pip install pulp


# In[94]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from datetime import datetime

import yfinance as yf


# In[95]:


#Get the stock starting date
stockStartDate = '2011-08-26'
# Get the stocks ending date aka todays date
#and format it in the form YYYY-MM-DD
today = datetime.today().strftime('%Y-%m-%d')


# In[78]:


assets='MRF.NS','BAJAJ-AUTO.NS','MUTHOOTFIN.NS','HDFC.NS','NESTLEIND.NS','FINPIPE.NS','NESCO.NS'


# In[96]:


assets='AMZN','TSLA'


# In[97]:


#Create a dataframe to store the adjusted close price of the stocks
df = pd.DataFrame()

#Store the adjusted close price of stock into the data frame
df=yf.download(assets,start=stockStartDate,end=today)['Adj Close']


# In[80]:


df.dropna(inplace=True,axis=0)


# # Create The Fictional Portfolio

# In[81]:


#Get the stock symbols / tickers for the fictional portfolio.
#assets='AAPL','CSIQ','GOOG','AMZN','MCD','MMM','MSFT','NFLX','TSLA','WMT'


# In[98]:


#assign equivalent weights to each stock within the portfolio
weights=np.array([0.2, 0.2])


# This means if I had a total of 100 EURO in the portfolio, then I would have 20 EURO in each stock.

# # Visulization for the stock prices

# In[99]:


# Create the title 'Portfolio Adj Close Price History
title = 'Portfolio Adj. Close Price History '
#Create and plot the graph
plt.figure(figsize=(12.2,4.5)) #width = 12.2in, height = 4.5# Loop through each stock and plot the Adj Close for each day
for c in df.columns.values:
    plt.plot( df[c],  label=c)
#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price EURO',fontsize=18)
plt.legend(df.columns.values, loc='upper left')
plt.show()


# In[100]:


#Show the daily simple returns, NOTE: Formula = new_price/old_price - 1
returns = df.pct_change()
returns


# Create and show the annualized co-variance matrix. 
# multiply the co-variance matrix by the number of trading days for the current year. In this case the number of trading days will be 252 for this year.

# In[101]:


cov_matrix_annual = returns.cov() * 252
cov_matrix_annual


# Now calculate and show the portfolio variance using the formula :
# Expected portfolio variance= WT * (Covariance Matrix) * W

# In[102]:


port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance


# Calculate the portfolio volatility using the formula :
# 
# Expected portfolio volatility= SQRT (WT * (Covariance Matrix) * W)

# In[103]:


port_volatility = np.sqrt(port_variance)
port_volatility


# Calculate the portfolio annual simple return.

# In[104]:


portfolioSimpleAnnualReturn = np.sum(returns.mean()*weights) * 252
portfolioSimpleAnnualReturn


# Show the expected annual return, volatility or risk, and variance.

# In[105]:


percent_var = str(round(port_variance, 2) * 100) + '%'
percent_vols = str(round(port_volatility, 2) * 100) + '%'
percent_ret = str(round(portfolioSimpleAnnualReturn, 2)*100)+'%'
print("Expected annual return : "+ percent_ret)
print('Annual volatility/standard deviation/risk : '+percent_vols)
print('Annual variance : '+percent_var)


# # Optimize The Portfolio

# Lets optimize the portfolio for maximum return with the least amount of risk . Luckily their is a very nice package that can help with this created by Robert Ansrew Martin.

# In[106]:


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


# Calculate the expected returns and the annualised sample covariance matrix of daily asset returns.

# In[107]:



mu = expected_returns.mean_historical_return(df)#returns.mean() * 252
S = risk_models.sample_cov(df) #Get the sample covariance matrix


# Optimize for maximal Sharpe ration .

# In[108]:


ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
#Maximize the Sharpe ratio, and get the raw weights
cleaned_weights = ef.clean_weights() 
print(cleaned_weights)
#Note the weights may have some rounding error, meaning they may not add up exactly to 1 but should be close
ef.portfolio_performance(verbose=True)


# Now we see that we can optimize this portfolio by having about 44.29% of the portfolio in Tesla , 55.71% in Amazon.

# Also I can see that the expected annual volatility has increased to 34.5% but the annual expected rate also to 50.7% is . This optimized portfolio has a Sharpe ratio of 1.41 which is good.

# I want to get the discrete allocation of each share of the stock, meaning I want to know exactly how many of each stock I should buy given some amount that I am willing to put into this portfolio.

# In[111]:


from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
latest_prices = get_latest_prices(df)
weights = cleaned_weights 
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: EURO {:.2f}".format(leftover))


# Alright ! Looks like I can buy  2 shares of Amazon and 4 shares of TSLA for this optimized portfolio and still have about 380.28 EURO leftover from my initial investment of 10000 EURO.

# In[ ]:





# In[ ]:





# In[ ]:




