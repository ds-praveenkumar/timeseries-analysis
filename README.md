# timeseries-analysis
data analysis related to Time series forecasting and prediction

## Problem Statement 

NOTE:
`data` folder needs to be added which must contain all the columns ['Date', 'Open', 'High', 'Low', 'Close', 'Shares Traded', 'Turnover (Rs. Cr)']

* data is be publically available in NSE India website.


## Folder Structure
- `data` - timeseries data for Bank Nifty from jan 2016 - Aug 2020
- `bank_nifty_forecasting` - code related to bank nifty forecasting 
- `bank_nifty_forecasting/build_model.py` - load model and prepare prediction
- `bank_nifty_forecasting/data_prep.py` - transforms data for training model

## Environment Setup
- `cd timeseries-forecasting`
- `pip install -r requirements.txt`


