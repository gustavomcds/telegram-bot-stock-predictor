import json

AVAILABLE_TICKERS_FILENAME = 'available_tickers'

stocks = ['SULA11.SA', 'ITSA4.SA', 'TOTS3.SA', 'BBAS3.SA', 'MGLU3.SA', 'ENBR3.SA', 'PRIO3.SA', 'TAEE11.SA']
reits = ['VGIP11.SA', 'IRDM11.SA', 'HGLG11.SA', 'BTLG11.SA', 'MXRF11.SA', 'KNRI11.SA']
cryptos = ['BTC-USD', 'ETH-USD', 'SOL1-USD', 'ADA-USD', 'DOT1-USD', 'ALGO-USD', 'MANA-USD', 'VET-USD']

available_tickers = {
    'stocks': stocks, 
    'reits': reits,
    'cryptos': cryptos
}

tickers = stocks + reits + cryptos

with open(f'./assets/{AVAILABLE_TICKERS_FILENAME}.json', 'w') as available_tickers_file:
    json.dump(available_tickers, available_tickers_file)