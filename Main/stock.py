import sys
import tkinter
from tkinter import *
from tkinter import messagebox
import tkinter as tk 
from tkinter import ttk
from tkinter import Canvas
from PIL import Image
from PIL import ImageTk,Image 
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pandas_datareader as web #Sub package that allows one to create a dataframe from various internet datasources, currently including: Yahoo! Finance. Google Finance.
import matplotlib.pyplot as plt # For plotting of statistical data
import datetime as dt # For generating real time date & time
import math
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

r = tk.Tk() 
r.title('Stock Market Management System')
r.configure(background="white")



def portfolio():
    #the datasets of stocks
    #Creating the portfolio of stocks owned with various companies
    tickers=['AAPL','FB','AMZN','TSLA','GOOG']
    amounts=[1, 1, 1, 1, 1]
    prices=[]
    total=[]

    for tickerr in tickers:
        df =web.DataReader(tickerr, 'yahoo')
        price = df[-1:]['Close'][0]
        prices.append(price)
        index = tickers.index(tickerr)
        total.append(price * amounts[index])

    #Visualizing portfolio
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_facecolor('black')
    ax.figure.set_facecolor('#121212')
    ax.tick_params(axis='x',color='white')
    ax.tick_params(axis='y',color='white')
    ax.set_title("PORTFOLIO VISUALIZER", color='#EF6C35', fontsize=20)
    patches, texts, autotexts = ax.pie(total, labels=tickers, autopct="%1.1f%%", pctdistance=0.8)
    [text.set_color('white') for text in texts]

    my_circle = plt.Circle((0,0), 0.55, color='black')
    plt.gca().add_artist(my_circle)

    ax.text(-2, 1, 'PORTFOLIO OVERVIEW', fontsize=14, color='#FFE356', verticalalignment='center', horizontalalignment='center')
    ax.text(-2, 0.85, f'TotalUSD Amount: {sum(total):.2f} $',  fontsize=12, color='white', verticalalignment='center', horizontalalignment='center')
    counter=0.15

    for tickerr in tickers:
        ax.text(-2, 0.85-counter, f'{tickerr}: {total[(tickers.index(tickerr))]:.2f} $', fontsize=12, color='white', verticalalignment='center', horizontalalignment='center')
        counter+=0.15
    plt.show()

def google():
    #Plotting the RSI comparison of google stocks
    ticker1='GOOG'
    data = web.DataReader(ticker1, 'yahoo') # Get the stocks details of company
    #print(data)

    delta = data['Adj Close'].diff(1) #Calculte the difference of the day before
    delta.dropna(inplace= True) #Get rid of non-number values

    positive = delta.copy() #Save the positive movements
    negative = delta.copy() #Save thennegative movements

    positive[positive < 0] = 0 #Either have positive or zero values
    negative[negative > 0] = 0 #Either have negative or zero values

    # Specify  a particular time frame
    days = 14

    # Calculate tow important values for RSI (Relative Strength Index)
    # Average gain
    # Average loss

    average_gain = positive.rolling(window=days).mean() #calculate the mean by taking the aggregate of past 14 days
    average_loss = abs(negative.rolling(window=days).mean())  #calculate the mean by taking the aggregate of past 14 days

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    #Create a combined dataframe

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # Plotting the figure

    #Plotting for adjusted close price

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color = 'lightgrey')
    ax1.grid(True, color='#eef2eb')
    ax1.set_axisbelow(True)
    ax1.set_title("Google: Adjusted Close Price", color='white')
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')


    #Plotting for RSI

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgrey')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_title("Google: RSI Value ",color= 'white')
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    plt.show()

def facebook():
    #Plotting the RSI comparison of facebook stocks
    ticker2='FB'
    data = web.DataReader(ticker2, 'yahoo') # Get the stocks details of company
    #print(data)

    delta = data['Adj Close'].diff(1) #Calculte the difference of the day before
    delta.dropna(inplace= True) #Get rid of non-number values

    positive = delta.copy() #Save the positive movements
    negative = delta.copy() #Save thennegative movements

    positive[positive < 0] = 0 #Either have positive or zero values
    negative[negative > 0] = 0 #Either have negative or zero values

    # Specify  a particular time frame
    days = 14

    # Calculate tow important values for RSI (Relative Strength Index)
    # Average gain
    # Average loss

    average_gain = positive.rolling(window=days).mean() #calculate the mean by taking the aggregate of past 14 days
    average_loss = abs(negative.rolling(window=days).mean())  #calculate the mean by taking the aggregate of past 14 days

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    #Create a combined dataframe

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # Plotting the figure

    #Plotting for adjusted close price

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color = 'lightgrey')
    ax1.grid(True, color='#eef2eb')
    ax1.set_axisbelow(True)
    ax1.set_title("Facebook: Adjusted Close Price", color='white')
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')
    #Plotting for RSI

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgrey')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_title("Facebook: RSI Value ",color= 'white')
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    plt.show()

def amazon():
    #Plotting for amazon
    ticker3='AMZN'
    data = web.DataReader(ticker3, 'yahoo') # Get the stocks details of company
    print(data)

    delta = data['Adj Close'].diff(1) #Calculte the difference of the day before
    delta.dropna(inplace= True) #Get rid of non-number values

    positive = delta.copy() #Save the positive movements
    negative = delta.copy() #Save thennegative movements

    positive[positive < 0] = 0 #Either have positive or zero values
    negative[negative > 0] = 0 #Either have negative or zero values

    # Specify  a particular time frame
    days = 14

    # Calculate tow important values for RSI (Relative Strength Index)
    # Average gain
    # Average loss

    average_gain = positive.rolling(window=days).mean() #calculate the mean by taking the aggregate of past 14 days
    average_loss = abs(negative.rolling(window=days).mean())  #calculate the mean by taking the aggregate of past 14 days

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    #Create a combined dataframe

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # Plotting the figure

    #Plotting for adjusted close price

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color = 'lightgrey')
    ax1.grid(True, color='#eef2eb')
    ax1.set_axisbelow(True)
    ax1.set_title("AMAZON: Adjusted Close Price", color='white')
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')


    #Plotting for RSI

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgrey')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_title("AMAZON: RSI Value ",color= 'white')
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    plt.show()

def tesla():
    #Plotting for Tesla
    ticker4='TSLA'
    data = web.DataReader(ticker4, 'yahoo') # Get the stocks details of company
    #print(data)

    delta = data['Adj Close'].diff(1) #Calculte the difference of the day before
    delta.dropna(inplace= True) #Get rid of non-number values

    positive = delta.copy() #Save the positive movements
    negative = delta.copy() #Save thennegative movements

    positive[positive < 0] = 0 #Either have positive or zero values
    negative[negative > 0] = 0 #Either have negative or zero values

    # Specify  a particular time frame
    days = 14

    # Calculate tow important values for RSI (Relative Strength Index)
    # Average gain
    # Average loss

    average_gain = positive.rolling(window=days).mean() #calculate the mean by taking the aggregate of past 14 days
    average_loss = abs(negative.rolling(window=days).mean())  #calculate the mean by taking the aggregate of past 14 days

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    #Create a combined dataframe

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # Plotting the figure

    #Plotting for adjusted close price

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color = 'lightgrey')
    ax1.grid(True, color='#eef2eb')
    ax1.set_axisbelow(True)
    ax1.set_title("Tesla: Adjusted Close Price", color='white')
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')


    #Plotting for RSI

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgrey')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_title("Tesla: RSI Value ",color= 'white')
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    plt.show()

def apple():
    #Plotting for Apple
    ticker5='AAPL'
    data = web.DataReader(ticker5, 'yahoo') # Get the stocks details of company
    #print(data)

    delta = data['Adj Close'].diff(1) #Calculte the difference of the day before
    delta.dropna(inplace= True) #Get rid of non-number values

    positive = delta.copy() #Save the positive movements
    negative = delta.copy() #Save thennegative movements

    positive[positive < 0] = 0 #Either have positive or zero values
    negative[negative > 0] = 0 #Either have negative or zero values

    # Specify  a particular time frame
    days = 14

    # Calculate tow important values for RSI (Relative Strength Index)
    # Average gain
    # Average loss

    average_gain = positive.rolling(window=days).mean() #calculate the mean by taking the aggregate of past 14 days
    average_loss = abs(negative.rolling(window=days).mean())  #calculate the mean by taking the aggregate of past 14 days

    relative_strength = average_gain / average_loss
    RSI = 100.0 - (100.0 / (1.0 + relative_strength))

    #Create a combined dataframe

    combined = pd.DataFrame()
    combined['Adj Close'] = data['Adj Close']
    combined['RSI'] = RSI

    # Plotting the figure

    #Plotting for adjusted close price

    plt.figure(figsize=(12,8))
    ax1 = plt.subplot(211)
    ax1.plot(combined.index, combined['Adj Close'], color = 'lightgrey')
    ax1.grid(True, color='#eef2eb')
    ax1.set_axisbelow(True)
    ax1.set_title("APPLE: Adjusted Close Price", color='white')
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('#121212')
    ax1.tick_params(axis = 'x', colors='white')
    ax1.tick_params(axis = 'y', colors='white')


    #Plotting for RSI

    ax2 = plt.subplot(212, sharex=ax1)
    ax2.plot(combined.index, combined['RSI'], color='lightgrey')
    ax2.axhline(0, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.axhline(10, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(20, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='#cccccc')
    ax2.axhline(80, linestyle='--', alpha=0.5, color='#00ff00')
    ax2.axhline(90, linestyle='--', alpha=0.5, color='#ffaa00')
    ax2.axhline(100, linestyle='--', alpha=0.5, color='#ff0000')
    ax2.grid(False)
    ax2.set_axisbelow(True)
    ax2.set_title("APPLE: RSI Value ",color= 'white')
    ax2.set_facecolor('black')
    ax2.tick_params(axis = 'x', colors='white')
    ax2.tick_params(axis = 'y', colors='white')
    plt.show()

Google = web.DataReader('GOOGL', 'yahoo')
Apple = web.DataReader('AAPL', 'yahoo')
Facebook = web.DataReader('FB', 'yahoo')
Amazon = web.DataReader('AMZN', 'yahoo')
Tesla = web.DataReader('TSLA', 'yahoo')
print("Google Stock Prices")
print(Google.head())
print("Apple Stock Prices")
print(Apple.head())
print("Facebook Stock Prices")
print(Facebook.head())
print("Amazon Stock Prices")
print(Amazon.head())
print("Tesla Stock Prices")
print(Tesla.head())

#Storing the stock files into database
Google.to_csv('Google_Stock.csv')
Apple.to_csv('Apple_Stock.csv')
Amazon.to_csv('Amazon_Stock.csv')
Facebook.to_csv('Facebook_Stock.csv')
Tesla.to_csv('Tesla_Stock.csv')

def google_stock():
    #Plotting Google stock prices
    Google['Open'].plot(label='Google Open Price',figsize=(16,8))
    Google['Low'].plot(label='Google Low Price')
    Google['Close'].plot(label='Google Close Price')
    Google['High'].plot(label='Google High Price')
    plt.title("Google Stock Prices")
    plt.xlabel('Dates')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()


def amazon_stock():
    #Plotting for Amazon
    Amazon['Open'].plot(label='Amazon Open Price',figsize=(16,8))
    Amazon['Low'].plot(label='Amazon Low Price')
    Amazon['Close'].plot(label='Amazon Close Price')
    Amazon['High'].plot(label='Amazon High Price')
    plt.title("Amazon Stock Prices")
    plt.xlabel('Dates')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

def facebook_stock():
    #Plotting for Facebook
    Facebook['Open'].plot(label='Facebook Open Price',figsize=(16,8))
    Facebook['Low'].plot(label='Facebook Low Price')
    Facebook['Close'].plot(label='Facebook Close Price')
    Facebook['High'].plot(label='Facebook High Price')
    plt.title("Facebook Stock Prices")
    plt.xlabel('Dates')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

def apple_stock():
    #Plotting for Apple
    Apple['Open'].plot(label='Apple Open Price',figsize=(16,8))
    Apple['Low'].plot(label='Apple Low Price')
    Apple['Close'].plot(label='Apple Close Price')
    Apple['High'].plot(label='Apple High Price')
    plt.title("Apple Stock Prices")
    plt.xlabel('Dates')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

def tesla_stock():
    #Plotting for Tesla
    Tesla['Open'].plot(label='Tesla Open Price',figsize=(16,8))
    Tesla['Low'].plot(label='Tesla Low Price')
    Tesla['Close'].plot(label='Tesla Close Price')
    Tesla['High'].plot(label='Tesla High Price')
    plt.title("Tesla Stock Prices")
    plt.xlabel('Dates')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()


def invest_on_return():
    Google['returns'] = (Google['Close']/Google['Close'].shift(1))-1  #Current day closing price - the day before losing price
    Apple['returns'] = (Apple['Close']/Apple['Close'].shift(1))-1  #Current day closing price - the day before losing price
    Amazon['returns'] = (Amazon['Close']/Amazon['Close'].shift(1))-1  #Current day closing price - the day before losing price
    Facebook['returns'] = (Facebook['Close']/Facebook['Close'].shift(1))-1  #Current day closing price - the day before losing price
    Tesla['returns'] = (Tesla['Close']/Tesla['Close'].shift(1))-1  #Current day closing price - the day before losing price
    #print(Google.head())

    # Calculating the cumulative returns
    """The cumulative return is the total change in the investment price over a set time—an aggregate return, not an annualized one. """

    Google['Cumulative Return'] = (1 + Google['returns']).cumprod()
    Amazon['Cumulative Return'] = (1 + Amazon['returns']).cumprod()
    Apple['Cumulative Return'] = (1 + Apple['returns']).cumprod()
    Facebook['Cumulative Return'] = (1 + Facebook['returns']).cumprod()
    Tesla['Cumulative Return'] = (1 + Tesla['returns']).cumprod()

    print("Cumulative Returns of Google")
    print(Google.head())
    print("Cumulative Returns of Amazon")
    print(Amazon.head())
    print("Cumulative Returns of Apple")
    print(Apple.head())
    print("Cumulative Returns of Facebook")
    print(Facebook.head())
    print("Cumulative Returns of Tesla")
    print(Tesla.head())

    Google['Cumulative Return'].plot(label='Google',figsize=(15,7))
    Amazon['Cumulative Return'].plot(label='Amazon')
    Apple['Cumulative Return'].plot(label='Apple')
    Facebook['Cumulative Return'].plot(label='Facebook')
    Tesla['Cumulative Return'].plot(label='Tesla')
    plt.title("Cumulative Return V/S Time", color='black')
    plt.legend()
    plt.show()

def linear_regression_google():

    df = pd.read_csv('Google_Stock.csv')
    #replace datafile.csv with your csv file

    print(df.head(25))

    x = df[['High','Open','Low','Close']].values
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    print(regressor.coef_)

    y_pred = regressor.predict(x_test)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    result.head(25)
    graph = result.tail(20)
    graph.plot(kind='bar')
    plt.show()

#facebook

def linear_regression_facebook():

    df = pd.read_csv('Facebook_Stock.csv')
    #replace datafile.csv with your csv file

    print(df.head(25))

    x = df[['High','Open','Low','Close']].values
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    print(regressor.coef_)

    y_pred = regressor.predict(x_test)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    result.head(25)
    graph = result.tail(20)
    graph.plot(kind='bar')
    plt.show()

#Amazon
def linear_regression_amazon():

    df = pd.read_csv('Amazon_Stock.csv')
    #replace datafile.csv with your csv file

    print(df.head(25))

    x = df[['High','Open','Low','Close']].values
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    print(regressor.coef_)

    y_pred = regressor.predict(x_test)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    result.head(25)
    graph = result.tail(20)
    graph.plot(kind='bar')
    plt.show()

#Applwe
def linear_regression_apple():

    df = pd.read_csv('Apple_Stock.csv')
    #replace datafile.csv with your csv file

    print(df.head(25))

    x = df[['High','Open','Low','Close']].values
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    print(regressor.coef_)

    y_pred = regressor.predict(x_test)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    result.head(25)
    graph = result.tail(20)
    graph.plot(kind='bar')
    plt.show()

#tesla
def linear_regression_tesla():

    df = pd.read_csv('Tesla_Stock.csv')
    #replace datafile.csv with your csv file

    print(df.head(25))

    x = df[['High','Open','Low','Close']].values
    y = df['Close'].values

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    regressor = LinearRegression()
    regressor.fit(x_train,y_train)

    print(regressor.coef_)

    y_pred = regressor.predict(x_test)
    result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
    result.head(25)
    graph = result.tail(20)
    graph.plot(kind='bar')
    plt.show()

#................................................................................................................................

#................................................................................................................................

#function to show stock prices
def stockprice():
    if ddl.get()=='Google_stock':
        google_stock()
    elif ddl.get()=='Amazon_stock':
        amazon_stock()
    elif ddl.get()=='Apple_stock':
        apple_stock()
    elif ddl.get()=='Facebook_stock':
        facebook_stock()
    elif ddl.get()=='Tesla_stock':
        tesla_stock()

#function to show RSI Values
def rsivalues():
    if dd2.get()=='Google':
        google()
    elif dd2.get()=='Facebook':
        facebook()
    elif dd2.get()=='Amazon':
        amazon()
    elif dd2.get()=='Tesla':
        tesla()
    elif dd2.get()=='Apple':
        apple()

#function to show actual precict
def predict():
    if dd3.get()=='Google_predict':
        linear_regression_google()
    elif dd3.get()=='Facebook_predict':
        linear_regression_facebook()
    elif dd3.get()=='Amazon_predict':
        linear_regression_amazon()
    elif dd3.get()=='Apple_predict':
        linear_regression_apple()
    elif dd3.get()=='Tesla_predict':
        linear_regression_tesla()



def exit():
    quit()
#System Name
hname= tk.Label(r, text="Stock Market",font=("Cursive",30),fg='red',bg='black')
hname.pack()
hname.place(x=450,y=60)
hname1= tk.Label(r, text="Management System",font=("Cursive",30),fg='sky blue',bg='black')
hname1.pack()
hname1.place(x=688,y=60)

#header
lbl1 = tk.Label(r, text="Company's Portfolio",fg='white',bg='black',width=20,font=("Cursive",16))
lbl1.pack()
lbl1.place(x=500,y=200)

lbl2 = tk.Label(r, text="RSI Values",fg='white',bg='black',width=25,font=("Cursive",16))
lbl2.pack()
lbl2.place(x=1000,y=300)

lbl3 = tk.Label(r, text="Actual/Predict",fg='white',bg='black',width=25,font=("Cursive",16))
lbl3.pack()
lbl3.place(x=1000,y=500)

lbl4 = tk.Label(r, text="Stock Prices",fg='white',bg='black',width=25,font=("Cursive",16))
lbl4.pack()
lbl4.place(x=250,y=300)

lbl5 = tk.Label(r, text="ROI Values",fg='white',bg='black',width=25,font=("Cursive",16))
lbl5.pack()
lbl5.place(x=250,y=500)

#code for menubar structure
btn2= tk.Button(r, text='Portfolio', width=26,font=("Cursive",13),command=portfolio) 
btn2.pack() 
btn2.place(x=750,y=199)

btn1= tk.Button(r, text='ROI',width=33,font=("Cursive",13),command=invest_on_return) 
btn1.pack() 
btn1.place(x=250,y=550)

ddl= ttk.Combobox(r,font=("Cursive",16),
                                values=[
                                        "Google_stock",
                                        "Amazon_stock", 
                                        "Apple_stock",
                                        "Facebook_stock",
                                        "Tesla_stock",
                                        ])
ddl.pack()
ddl.place(x=250,y=350,height=30,width=300)

btn5= tk.Button(r, text='View Stock Prices', width=33,font=("Cursive",13),command=stockprice) 
btn5.pack() 
btn5.place(x=250,y=390)


dd2= ttk.Combobox(r,font=("Cursive",16),
                                values=[
                                        "Google",
                                        "Facebook", 
                                        "Amazon",
                                        "Tesla",
                                        "Apple",
                                        ])
dd2.pack()
dd2.place(x=1000,y=350,height=30,width=300)

btn6= tk.Button(r, text='View ROI Values', width=33,font=("Cursive",13),command=rsivalues) 
btn6.pack() 
btn6.place(x=1000,y=390)

dd3= ttk.Combobox(r,font=("Cursive",16),
                                values=[
                                        "Google_predict",
                                        "Facebook_predict",
                                        "Amazon_predict",
                                        "Apple_predict",
                                        "Tesla_predict"
                                        ])
dd3.pack()
dd3.place(x=1000,y=550,height=30,width=300)

btn6= tk.Button(r, text='Predict', width=33,font=("Cursive",13),command=predict) 
btn6.pack() 
btn6.place(x=1000,y=590)

btn7= tk.Button(r, text='Exit', width=33,font=("Cursive",13),command=exit)
btn7.pack()
btn7.place(x=630,y=700)


r.geometry('1545x775')
r.mainloop()    
#................................................................................................................................

#................................................................................................................................
