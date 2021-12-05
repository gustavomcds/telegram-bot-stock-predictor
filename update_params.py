import json
from tqdm import tqdm
import yfinance as yf

from regressors.methods import optimize_holt_winters_hyperparameters

with open('./assets/available_tickers.json', 'r') as available_tickers_file:
    available_tickers = json.load(available_tickers_file)

tickers = available_tickers['stocks'] + available_tickers['reits'] + available_tickers['cryptos']

dict_best_params = dict()

for ticker in tqdm(tickers):
    
    yf_tckr = yf.Ticker(ticker)
    tckr_history = yf_tckr.history(period='5y', interval='1d').reset_index()
    tckr_history.fillna(method='ffill', inplace=True)
    ts = tckr_history['Close'].values
    
    best_hyperparams = optimize_holt_winters_hyperparameters(ts, slens=[90, 180, 365])
    
    dict_best_params[ticker] = best_hyperparams

with open('./assets/tes_params.json', 'w') as tes_params_file:
    json.dump(dict_best_params, tes_params_file, default=str)