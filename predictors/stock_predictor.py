# importing libraries

from regressors.methods import HoltWinters

import pandas as pd
import numpy as np
from itertools import product

from datetime import datetime, timedelta
import json

import yfinance as yf

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

# setting matplotlib params
plt.rcParams['figure.figsize'] = (18, 6)


class StockPredictor:
    
    def __init__(self, ticker, n_preds):
        
        self.ticker = ticker
        self.n_preds = n_preds
        self.past_dates = None
        self.future_dates = None
        
    def get_data(self, period='5y', interval='1d'):
        
        self.tckr = yf.Ticker(self.ticker)
        self.df_tckr = self.tckr.history(period=period, interval=interval).reset_index()
        self.df_tckr.fillna(method='ffill', inplace=True)
        self.ts_tckr = self.df_tckr['Close'].values
        
    def get_tes_params(self):
        
        with open('./assets/tes_params.json', 'r') as params_file:
            tes_params = json.load(params_file)
            
        self.alpha = float(tes_params[self.ticker]['alpha'])
        self.beta = float(tes_params[self.ticker]['beta'])
        self.gamma = float(tes_params[self.ticker]['gamma'])
        self.slen = int(tes_params[self.ticker]['slen'])
    
    def predict(self):

        self.past_dates = pd.to_datetime(self.df_tckr['Date']).values
        self.future_dates = np.array(pd.to_datetime(pd.date_range(start=self.past_dates[-1], periods=90, freq='B')))
        
        self.hw = HoltWinters(self.ts_tckr, self.slen, self.alpha, self.beta, self.gamma, self.n_preds, past_dates=self.past_dates, future_dates=self.future_dates)
        self.preds = self.hw.triple_exponential_smoothing()
        
    def build_plot_predictions(self):
        
        self.chart_pred = self.hw.plotHoltWinters(ticker_title=self.ticker)