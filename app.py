import requests
from pandas.io.json import json_normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import axis0_safe_slice

# classical Time Series Tools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARMA, ARIMA
import tensorflow as tf

# Start Code, get URL and display plot after cleansing data
url = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=ada&market=EUR&apikey=J2WB3Q124ZZC6WN4"
resp = requests.get(url=url)
data = resp.json()
cardano_df = pd.DataFrame(data["Time Series (Digital Currency Daily)"])
cardano_df = cardano_df.transpose()
cardano_df.sort_index(ascending=True,inplace=True)
_finalArr = []
# Iteration to clean the data, JSON Returns string, we need to convert to Float (I don't understand why really)
for index, row in cardano_df.iterrows():
    _arr = [index, float(row['4b. close (USD)']),float(row['5. volume']),float(row['3b. low (USD)']),float(row['2b. high (USD)']),float(row['1b. open (USD)'])]
    _finalArr.append(_arr)

cardano_df = pd.DataFrame(_finalArr, columns=['date','close','volume','low','high','open'])
cardano_df['SMA'] = cardano_df.iloc[:,1].rolling(window=31).mean()
cardano_df['Std'] = cardano_df.iloc[:,1].rolling(window=31).std()

# From the Data, split into Training data, Validation and Testing
length = len(cardano_df)

train_df = cardano_df[30:int(length*0.8)]
test_df  = cardano_df[int(0.8*length):]


# Declaring X  
train_Y = train_df[['close']]
test_Y  = test_df[['close']]

required_cols = ['high','low','open','volume','SMA','Std']
train_X = train_df[required_cols]
test_X  = test_df[required_cols]

from sklearn.linear_model import LinearRegression

lnr_reg = LinearRegression()
lnr_reg.fit(train_X,train_Y)

predictions = lnr_reg.predict(test_X)

test_Y.insert(0,'Predictions',predictions)
test_Y.plot()
plt.show()
