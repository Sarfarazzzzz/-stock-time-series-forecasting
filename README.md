
Stock Time Series Forecasting using ARMA, ARIMA, and Box–Jenkins Models
========================================================================

This repository contains the complete source code and supporting material for a comprehensive time series forecasting project on three Indian stock tickers: TATAMOTORS, TATASTEEL, and TCS. The project explores a wide range of statistical forecasting techniques including ARMA, ARIMA, and Box–Jenkins with exogenous input, and performs rigorous model diagnostics and residual analysis.

PROJECT OBJECTIVES:
-------------------
- Conduct time series analysis on daily stock data
- Perform decomposition and stationarity tests (ADF, KPSS)
- Build and evaluate ARMA, ARIMA, and Box–Jenkins models
- Incorporate external variables for enhanced forecasting
- Conduct full diagnostic testing (Ljung-Box, S-test, variance checks)
- Generate multi-step ahead forecasts and compare with test data

REPOSITORY STRUCTURE:
---------------------
- final code.py              : Main pipeline script for data preprocessing, modeling, diagnostics
- Toolkit.py                 : Helper functions for differencing, ADF/KPSS, GPAC, forecasting, etc.
- data/                      : Contains CSV files for TATAMOTORS, TATASTEEL, TCS
- figures/                   : Saved plots used in the report (ACF/PACF, forecast graphs, diagnostics)
- results/                   : Model performance outputs, metrics, and diagnostic logs
- README.txt                 : Project overview and instructions
- requirements.txt           : Required Python libraries
- report.pdf                 : Final academic report (if uploaded)

MODELS USED:
------------
- Base Models: Naïve, Drift, Average, Simple Exponential Smoothing
- ARMA/ARIMA: Parameter selection via GPAC and ACF/PACF
- Box–Jenkins with Exogenous Input: Developed using LM algorithm and G–GPAC/H–GPAC
- Forecasting: One-step and multi-step ahead forecasts evaluated with RMSE and residual tests

HOW TO RUN:
-----------
1. Clone the repository:
   git clone https://github.com/Sarfarazzzzz/stock-time-series-forecasting.git
   cd stock-time-series-forecasting

2. Install dependencies:
   pip install -r requirements.txt

3. Place data files in the /data/ folder with these names:
   - TATAMOTORS.csv
   - TATASTEEL.csv
   - TCS.csv

4. Run the code:
   python "final code.py"

All plots and outputs will be saved in the 'results/' and 'figures/' directories.

KEY OUTCOMES:
-------------
- Performed robust model selection using AIC, BIC, RMSE
- Box–Jenkins model outperformed base models with up to 93% improvement
- Residuals validated via Ljung–Box, S-test, and whiteness chi-square test
- Multi-step forecasts closely aligned with actual stock movements

LIMITATIONS:
------------
- Financial time series are noisy and influenced by macroeconomic events
- Box–Jenkins assumes linearity and stationarity in differenced series
- Consider using LSTM, Prophet, or regime-switching models in the future

REFERENCES:
-----------
1. Box, G. E. P., Jenkins, G. M., & Reinsel, G. C. (2015). Time Series Analysis: Forecasting and Control
2. Hamilton, J. D. (1994). Time Series Analysis
3. Statsmodels Documentation: https://www.statsmodels.org
4. Kaggle Dataset: Tata Stocks

APPENDIX:
---------
- Toolkit.py: https://github.com/Sarfarazzzzz/stock-time-series-forecasting/blob/main/Toolkit.py
- final code.py: https://github.com/Sarfarazzzzz/stock-time-series-forecasting/blob/main/final%20code.py

CONTACT:
--------
Mohammed Ismail Sarfaraz Shaik
Graduate Student, Data Science
The George Washington University
Email: m.shaik@gwu.edu
