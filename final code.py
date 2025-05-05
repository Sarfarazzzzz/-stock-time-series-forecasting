#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import statsmodels.api as sm
from numpy.linalg import inv
from scipy.stats import chi2
from statsmodels.tsa.holtwinters import Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from numpy.linalg import LinAlgError, pinv
from sklearn.decomposition import PCA
from scipy.stats import t as t_dist
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARMA
from Toolkit import *

#%%

symbol_mapping = {
    'TCS': 'TCS',
    'TELCO': 'TATAMOTORS',  # Map TELECO to TATAMOTORS
    'TATAMOTORS': 'TATAMOTORS',
    'TISCO': 'TATASTEEL',    # Map TISCO to TATASTEEL
    'TATASTEEL': 'TATASTEEL'
}

df_tcs = pd.read_csv('TCS.csv', parse_dates=['Date'], index_col='Date')
df_tatamotors = pd.read_csv('TATAMOTORS.csv', parse_dates=['Date'], index_col='Date')
df_tatasteel = pd.read_csv('TATASTEEL.csv', parse_dates=['Date'], index_col='Date')

df_tcs['Symbol'] = df_tcs['Symbol'].map(symbol_mapping)
df_tatamotors['Symbol'] = df_tatamotors['Symbol'].map(symbol_mapping)
df_tatasteel['Symbol'] = df_tatasteel['Symbol'].map(symbol_mapping)

combined_df = pd.concat([df_tcs, df_tatamotors, df_tatasteel])
combined_df.sort_index(inplace=True)

combined_df.info()

#%%
mask_pd = (combined_df['%Deliverble'].isna()) & (combined_df['Deliverable Volume'].notna()) & (
            combined_df['Volume'] != 0)
combined_df.loc[mask_pd, '%Deliverble'] = (combined_df.loc[mask_pd, 'Deliverable Volume'] / combined_df.loc[
    mask_pd, 'Volume']) * 100


mask_dv = (combined_df['Deliverable Volume'].isna()) & (combined_df['%Deliverble'].notna()) & (
            combined_df['Volume'] != 0)
combined_df.loc[mask_dv, 'Deliverable Volume'] = (combined_df.loc[mask_dv, 'Volume'] * combined_df.loc[
    mask_dv, '%Deliverble']) / 100


def fill_missing_in_group(group):

    median_pd = group['%Deliverble'].median()


    mask_both = group['%Deliverble'].isna() & group['Deliverable Volume'].isna()


    group.loc[mask_both, '%Deliverble'] = median_pd


    group.loc[mask_both, 'Deliverable Volume'] = (group.loc[mask_both, 'Volume'] * median_pd) / 100

    return group

combined_df = combined_df.groupby('Symbol').apply(fill_missing_in_group)
combined_df.info()

#%%
valid_trades = combined_df['Trades'].notna() & (combined_df['Trades'] > 0)
combined_df.loc[valid_trades, 'Avg_Trade_Size'] = combined_df.loc[valid_trades, 'Volume'] / combined_df.loc[valid_trades, 'Trades']

median_avg_trade_size = combined_df.loc[valid_trades, 'Avg_Trade_Size'].median()
print("Median Average Trade Size:", median_avg_trade_size)

missing_trades = combined_df['Trades'].isna()

missing_trades &= combined_df['Volume'] > 0
combined_df.loc[missing_trades, 'Trades'] = combined_df.loc[missing_trades, 'Volume'] / median_avg_trade_size

combined_df.drop(columns=['Avg_Trade_Size'], inplace=True)

combined_df.info()


#%%
# label encoding

combined_df['Symbol_encoded'] = combined_df['Symbol'].astype('category').cat.codes
print(combined_df[['Symbol', 'Symbol_encoded']].head())
print(combined_df.head())

#%%

plt.figure(figsize=(12, 6))

# Loop through each unique stock symbol
for symbol in combined_df.index.get_level_values('Symbol').unique():
    # Extract the data for the given symbol using xs (cross-section)
    stock_data = combined_df.xs(symbol, level='Symbol')
    # Plot the 'Close' price against the index (which should now be Date)
    plt.plot(stock_data.index, stock_data['Close'], lw=2, label=symbol)

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Close Price Over Time')
plt.legend()
plt.grid(True)
plt.show()

#%%

df_for_corr = combined_df.copy()
df_for_corr = df_for_corr.drop(columns=['Symbol', 'Symbol_encoded', 'Series'])
correlation_matrix = df_for_corr.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title("Pearson Correlation Matrix", fontsize=16)
plt.tight_layout()

plt.show()


#%%

cols_to_drop = ['Prev Close', 'High', 'Low', 'Last', 'VWAP', 'Series']
combined_df.drop(columns=cols_to_drop, inplace=True)


#%%
df_for_corr2 = combined_df.copy()
df_for_corr2 = df_for_corr2.drop(columns=['Symbol', 'Symbol_encoded'])
correlation_matrix = df_for_corr2.corr()

plt.figure(figsize=(12, 10))

sns.heatmap(correlation_matrix,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title("Pearson Correlation Matrix", fontsize=16)
plt.tight_layout()

plt.show()

#%%

X = combined_df.drop(columns=['Close'])
y = combined_df['Close']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#%%

stocks = combined_df.index.get_level_values('Symbol').unique()
rolling_window = 30

fig, axes = plt.subplots(len(stocks), 1, figsize=(12, 4 * len(stocks)), sharex=True)

for i, stock in enumerate(stocks):

    stock_data = combined_df.xs(stock, level='Symbol')
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(level='Symbol', drop=True)

    ts = stock_data['Close']
    rolling_mean = ts.rolling(window=rolling_window).mean()
    rolling_var = ts.rolling(window=rolling_window).var()

    ax = axes[i] if len(stocks) > 1 else axes
    ax.plot(ts.index, ts, label='Raw Close', color='blue', lw=1.5)
    ax.plot(rolling_mean.index, rolling_mean, label=f'Rolling Mean', color='red')
    ax.plot(rolling_var.index, rolling_var, label=f'Rolling Var', color='green')
    ax.set_title(f'{stock} - Rolling Mean & Var')
    ax.legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%%

for stock in stocks:

    stock_data = combined_df.xs(stock, level='Symbol')
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(level='Symbol', drop=True)
    ts = stock_data['Close']

    adf_result = adfuller(ts)
    kpss_result = kpss(ts, regression='c', nlags='auto')

    print(f"----- {stock} - Stationarity Test Results (Raw Close) -----")
    print(f"ADF Statistic: {adf_result[0]:.3f}")
    print(f"ADF p-value: {adf_result[1]:.3f}")
    print("ADF Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.3f}")

    print(f"\nKPSS Statistic: {kpss_result[0]:.3f}")
    print(f"KPSS p-value: {kpss_result[1]:.3f}")
    print("KPSS Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value:.3f}")
    print("\n")

#%%

stocks = combined_df.index.get_level_values('Symbol').unique()

fig, axes = plt.subplots(nrows=len(stocks), ncols=2, figsize=(16, 4 * len(stocks)), sharex=True)

for i, stock in enumerate(stocks):

    stock_data = combined_df.xs(stock, level='Symbol')
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(level='Symbol', drop=True)

    ts = stock_data['Close']

    plot_acf(ts, lags=50, ax=axes[i, 0])
    axes[i, 0].set_title(f'{stock} - ACF')

    plot_pacf(ts, lags=50, ax=axes[i, 1])
    axes[i, 1].set_title(f'{stock} - PACF')

plt.tight_layout()
plt.show()

#%%

combined_df['Close_diff'] = combined_df.groupby(level='Symbol')['Close'].diff()

diff_df = combined_df.dropna(subset=['Close_diff'])
print(diff_df['Close_diff'].head())
print(diff_df['Close_diff'].tail())

#%%
if 'Symbol' in diff_df.columns:
    diff_df = diff_df.drop(columns=['Symbol'])
df_reset = diff_df.reset_index()
df_reset[['Date', 'Symbol', 'Close', 'Close_diff']].to_csv("filtered_stock_data.csv", index=False)

#%%

if 'Symbol' in diff_df.columns:
    diff_df = diff_df.drop(columns=['Symbol'])
df_reset = diff_df.reset_index()

stocks = df_reset['Symbol'].unique()
rolling_window = 30

fig, axes = plt.subplots(len(stocks), 1, figsize=(12, 4 * len(stocks)), sharex=True)

for i, stock in enumerate(stocks):

    stock_data = df_reset[df_reset['Symbol'] == stock].copy()

    stock_data = stock_data.set_index('Date')

    ts_diff = stock_data['Close_diff']
    ts_diff = ts_diff.replace([np.inf, -np.inf], np.nan).dropna()

    rolling_mean_diff = ts_diff.rolling(window=rolling_window).mean()
    rolling_var_diff = ts_diff.rolling(window=rolling_window).var()

    ax = axes[i] if len(stocks) > 1 else axes

    ax.plot(rolling_mean_diff.index, rolling_mean_diff, label='Rolling Mean', color='red')
    ax.plot(rolling_var_diff.index, rolling_var_diff, label='Rolling Variance', color='green')
    ax.set_title(f'{stock} - Rolling Mean & Variance (Differenced)')
    ax.legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()

#%%

for stock in stocks:

    stock_data = diff_df.xs(stock, level='Symbol')
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(level='Symbol', drop=True)
    ts = stock_data['Close_diff']


    adf_result = adfuller(ts)
    kpss_result = kpss(ts, regression='c', nlags='auto')


    print(f"----- {stock} - Stationarity Test Results (Raw Close) -----")
    print(f"ADF Statistic: {adf_result[0]:.3f}")
    print(f"ADF p-value: {adf_result[1]:.3f}")
    print("ADF Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.3f}")

    print(f"\nKPSS Statistic: {kpss_result[0]:.3f}")
    print(f"KPSS p-value: {kpss_result[1]:.3f}")
    print("KPSS Critical Values:")
    for key, value in kpss_result[3].items():
        print(f"   {key}: {value:.3f}")
    print("\n")

#%%

stocks = df_reset['Symbol'].unique()

for stock in stocks:

    stock_data = df_reset[df_reset['Symbol'] == stock].copy()
    stock_data = stock_data.set_index('Date')

    ts_diff = stock_data['Close_diff']
    ts_diff = ts_diff.replace([np.inf, -np.inf], np.nan).dropna()

    plt.figure(figsize=(14, 5))
    plt.plot(stock_data.index, stock_data['Close'], label='Original Close', color='blue', linewidth=1.5)
    plt.plot(stock_data.index, stock_data['Close_diff'], label='Differenced Close', color='orange', linewidth=1)

    plt.title(f'{stock} - Original vs Differenced Close Price')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%

stocks = diff_df.index.get_level_values('Symbol').unique()

fig, axes = plt.subplots(nrows=len(stocks), ncols=2, figsize=(16, 4 * len(stocks)), sharex=True)

for i, stock in enumerate(stocks):

    stock_data = diff_df.xs(stock, level='Symbol')
    if isinstance(stock_data.index, pd.MultiIndex):
        stock_data = stock_data.reset_index(level='Symbol', drop=True)

    ts_diff = stock_data['Close_diff']
    ts_diff = ts_diff.replace([np.inf, -np.inf], np.nan).dropna()
    print(acorr_ljungbox(ts_diff, lags=[10], return_df=True))

    plot_acf(ts_diff, lags=20, ax=axes[i, 0])
    axes[i, 0].set_title(f'{stock} - ACF')

    plot_pacf(ts_diff, lags=20, ax=axes[i, 1])
    axes[i, 1].set_title(f'{stock} - PACF')

plt.tight_layout()
plt.show()

#%%

for stock in df_reset['Symbol'].unique():
    print(f"\n STL Decomposition for: {stock}")

    stock_data = df_reset[df_reset['Symbol'] == stock].copy()
    stock_data = stock_data.set_index('Date')
    stock_data = stock_data.sort_index()

    ts = stock_data['Close'].replace([np.inf, -np.inf], np.nan).dropna()

    stl_add = STL(ts, period=30, seasonal=13, robust=True).fit()

    ts_log = np.log(ts)
    stl_mul = STL(ts_log, period=30, seasonal=13, robust=True).fit()

    resid_add = stl_add.resid
    trend_add = stl_add.trend
    seasonal_add = stl_add.seasonal

    var_resid = np.var(resid_add)
    var_trend = np.var(trend_add + resid_add)
    var_seasonal = np.var(seasonal_add + resid_add)

    trend_strength = max(0, 1 - (var_resid / var_trend))
    seasonal_strength = max(0, 1 - (var_resid / var_seasonal))

    print(f"Trend Strength (Additive): {trend_strength:.4f}")
    print(f"Seasonality Strength (Additive): {seasonal_strength:.4f}")

    fig_add = stl_add.plot()
    fig_add.suptitle(f'{stock} - STL (Additive)', fontsize=14)
    plt.tight_layout()
    plt.show()

    fig_mul = stl_mul.plot()
    fig_mul.suptitle(f'{stock} - STL (Multiplicative, Log)', fontsize=14)
    plt.tight_layout()
    plt.show()

#%%

df = df_reset.copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
dfs = []
for sym in df['Symbol'].unique():
    ts = df[df['Symbol']==sym]['Close'].sort_index().asfreq('B').ffill()
    dfs.append(ts.to_frame().assign(Symbol=sym))
df = pd.concat(dfs)

h = 30
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

methods = {
    'Average':   lambda tr: np.repeat(tr.mean(), h),
    'Naïve':     lambda tr: np.repeat(tr.iloc[-1], h),
    'Drift':     lambda tr: np.array([tr.iloc[-1] + (i+1)*(tr.iloc[-1]-tr.iloc[0])/(len(tr)-1) for i in range(h)]),
    'ExpSmooth': lambda tr: SimpleExpSmoothing(tr).fit().forecast(h).values
}

results = []

for sym in df['Symbol'].unique():
    ts = df[df['Symbol']==sym]['Close']
    train, test = ts[:-h], ts[-h:]
    future_idx = pd.date_range(start=test.index[-1]+pd.Timedelta(days=1),
                               periods=h, freq='B')

    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 4*len(methods)), sharex=True)
    for ax, (name, fn) in zip(axes, methods.items()):
        vals = fn(train)
        fc = pd.Series(vals, index=future_idx)
        err = rmse(test.values, vals)

        results.append({'Symbol': sym, 'Method': name, 'RMSE': err})

        ax.plot(train, label='Train', color='blue')
        ax.plot(test,  label='Test',  color='black')
        style = '-' if name=='Drift' else '--'
        ax.plot(fc, style, color='red', label=f'{name} Forecast')
        zoom_days = 60
        zs = train.index[-1] - pd.Timedelta(days=zoom_days)
        ze = future_idx[-1] + pd.Timedelta(days=5)
        ax.set_xlim(zs, ze)
        ax.set_title(f'{sym} — {name} (h={h}) RMSE={err:.2f}')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

results_df = pd.DataFrame(results)
pivot = results_df.pivot(index='Symbol', columns='Method', values='RMSE')
print(pivot.round(2))

#%%

df = df_reset.copy()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

dfs = []
for sym in df['Symbol'].unique():
    ts = (df[df['Symbol'] == sym]['Close']
          .sort_index()
          .asfreq('B')
          .ffill())
    dfs.append(ts.to_frame().assign(Symbol=sym))
df = pd.concat(dfs)

def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

h = 30

symbols = df['Symbol'].unique()
fig, axes = plt.subplots(len(symbols), 1, figsize=(12, 5 * len(symbols)), sharex=True)

rmse_values = {}

for ax, sym in zip(axes, symbols):
    ts = df[df['Symbol'] == sym]['Close']
    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    model = ExponentialSmoothing(train, trend='add', damped_trend=True, initialization_method='estimated')
    fit = model.fit(optimized=True)
    fc = fit.forecast(len(test))
    fc.index = test.index

    error = rmse(test.values, fc.values)
    rmse_values[sym] = error

    ax.plot(train, label='Train', color='tab:blue')
    ax.plot(test, label='Test', color='tab:orange')
    ax.plot(fc, '--', label=f"Damped Holt Forecast", color='tab:green')
    ax.set_title(f"{sym} — RMSE = {error:.2f}", fontsize=14)
    ax.set_ylabel("Close Price", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Date", fontsize=12)
plt.tight_layout()
plt.show()

print("Damped Holt Forecast RMSEs:")
for sym, val in rmse_values.items():
    print(f"{sym}: RMSE = {val:.2f}")

#%%

df = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])
df = df.dropna()
df = df.sort_values(["Symbol", "Date"])

df_raw = pd.concat([
    pd.read_csv("TCS.csv", parse_dates=["Date"]),
    pd.read_csv("TATAMOTORS.csv", parse_dates=["Date"]),
    pd.read_csv("TATASTEEL.csv", parse_dates=["Date"]),
], ignore_index=True)

df_raw = df_raw[["Date", "Symbol", "Volume", "Turnover", "Trades", "Deliverable Volume", "%Deliverble"]]
df = pd.merge(df, df_raw, on=["Date", "Symbol"], how="inner")

diff_cols = ["Volume", "Turnover", "Trades", "Deliverable Volume", "%Deliverble"]
df = df.sort_values(["Symbol", "Date"])
for col in diff_cols:
    df[f"{col}_diff"] = df.groupby("Symbol")[col].diff()

df = df.dropna(subset=[f"{col}_diff" for col in diff_cols] + ["Close_diff"])

def backward_elimination_vif(X, y, alpha=0.05, vif_thresh=15):
    Xc = sm.add_constant(X)
    while True:
        model = sm.OLS(y, Xc).fit()
        vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])], index=Xc.columns)
        if vif.drop("const").max() > vif_thresh:
            drop_feat = vif.drop("const").idxmax()
            print(f"Dropping '{drop_feat}' due to high VIF = {vif[drop_feat]:.2f}")
            Xc = Xc.drop(columns=[drop_feat])
            continue
        pvals = model.pvalues.drop("const")
        if pvals.max() > alpha:
            drop_p = pvals.idxmax()
            print(f"Dropping '{drop_p}' due to high p-value = {pvals[drop_p]:.4f}")
            Xc = Xc.drop(columns=[drop_p])
            continue
        break
    return sm.OLS(y, Xc).fit()

for sym in df["Symbol"].unique():
    print(f"\n========= {sym} =========")
    df_sym = df[df["Symbol"] == sym].copy()
    feature_cols = [f"{c}_diff" for c in diff_cols]
    X = df_sym[feature_cols].astype(float)
    y = df_sym["Close_diff"].astype(float)

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    X_train_c = sm.add_constant(X_train_s)
    X_test_c = sm.add_constant(X_test_s)

    model = backward_elimination_vif(X_train_s, y_train)
    print(model.summary())

    history_X = X_train_c.copy()
    history_y = y_train.copy()
    preds = []
    for i in range(len(X_test_c)):
        m = sm.OLS(history_y, history_X).fit()
        x_i = X_test_c.iloc[[i]]
        pred = m.predict(x_i).iloc[0]
        preds.append(pred)
        history_X = pd.concat([history_X, x_i])
        history_y = pd.concat([history_y, pd.Series([y_test.iloc[i]], index=[x_i.index[0]])])

    mse = mean_squared_error(y_test, preds)
    rmse_val = np.sqrt(mse)
    resid = model.resid
    resid_mean = np.mean(resid)
    resid_var = np.var(resid)
    aic = model.aic
    bic = model.bic
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    lbq = acorr_ljungbox(resid, lags=[10], return_df=True)

    print("\n--- Regression Metrics ---")
    print(f"MSE: {mse:.4f}, RMSE: {rmse_val:.4f}")
    print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
    print(f"R²: {r2:.4f}, Adjusted R²: {adj_r2:.4f}")
    print(f"Residual Mean: {resid_mean:.4e}, Variance: {resid_var:.4f}")
    print("Ljung–Box Q-Test:\n", lbq)

    plt.figure(figsize=(6, 3))
    plot_acf(resid, lags=20)
    plt.title(f"{sym} — ACF of Residuals")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(y_train.index, y_train, label="Train ΔClose", color="blue")
    plt.plot(y_test.index, y_test, label="Test ΔClose", color="black")
    plt.plot(y_test.index, preds, "--", label="Predicted ΔClose", color="red")
    plt.title(f"{sym} — One-Step-Ahead Forecast")
    plt.xlabel("Date")
    plt.ylabel("ΔClose")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%

df = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])

def gpac_table(acf_vals, max_k=10, max_j=10, model_nb=None):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            if model_nb is not None and j >= model_nb + 1:
                gpac[j, k - 1] = np.nan
                continue
            try:
                D = np.zeros((k, k))
                for i in range(k):
                    for m in range(k):
                        lag = abs(j + i - m)
                        D[i, m] = acf_vals[lag] if lag < len(acf_vals) else 0
                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf_vals[lag] if lag < len(acf_vals) else 0
                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)
                if np.isclose(det_D, 0):
                    if np.isclose(det_N, 0):
                        gpac[j, k - 1] = np.nan
                    else:
                        gpac[j, k - 1] = np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D
            except Exception:
                gpac[j, k - 1] = np.nan
    df_out = pd.DataFrame(
        gpac,
        index=[f"j={j}" for j in range(max_j)],
        columns=[f"k={k+1}" for k in range(max_k)]
    ).round(3)
    return df_out

def plot_gpac_table(gpac_df, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, fmt=".3f", cmap="rocket_r", cbar=True, vmin=-2, vmax=2)
    plt.title(f"GPAC Table — {title}", fontsize=14)
    plt.xlabel("k (AR order)")
    plt.ylabel("j (MA order)")
    plt.tight_layout()
    plt.show()

for symbol in df['Symbol'].unique():
    print(f"\n====== GPAC Table for {symbol} ======")
    stock_df = df[df['Symbol'] == symbol].copy()
    ts = stock_df['Close_diff'].replace([np.inf, -np.inf], np.nan).dropna()

    acf_vals = acf(ts, nlags=50, fft=True)
    gpac_df = gpac_table(acf_vals, max_k=10, max_j=10)
    print(gpac_df)

    plot_gpac_table(gpac_df, title=symbol)

#%%

df = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])
df = df.replace([np.inf, -np.inf], pd.NA).dropna()

stocks = df['Symbol'].unique()

for stock in stocks:
    ts = df[df['Symbol'] == stock].sort_values("Date")["Close_diff"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"{stock} - ACF and PACF", fontsize=16)

    plot_acf(ts, ax=axes[0], lags=30)
    plot_pacf(ts, ax=axes[1], lags=30)

    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.show()

#%%

'''
df = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])
df = df[["Date", "Symbol", "Close_diff", "Close"]].dropna()

# Ensure sorting
df = df.sort_values(by=["Symbol", "Date"])

# Set model orders
orders = [(2, 2),(1,1)]
symbols = df["Symbol"].unique()

# Fit models for each stock and each order
for sym in symbols:
    print(f"\n====== {sym} ======")
    stock_df = df[df["Symbol"] == sym].copy()
    stock_df.set_index("Date", inplace=True)
    ts = stock_df["Close_diff"].replace([np.inf, -np.inf], np.nan).dropna()

    for order in orders:
        p, q = order
        print(f"--- {sym} ARMA({p},{q}) on ΔClose ---")
        try:
            model = SARIMAX(ts, order=(p, 0, q), enforce_stationarity=False, enforce_invertibility=False)
            result = model.fit(disp=False)

            print(result.summary())
            print("params :", result.params.values)
            print("stderr :", result.bse.values)
            print("95% CI:\n", result.conf_int())
        except Exception as e:
            print(f"Failed to fit ARMA({p},{q}) for {sym}: {e}")'''

#%%

df_diff = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])
df_diff = df_diff[["Date", "Symbol", "Close_diff"]].dropna()

df_diff = df_diff.sort_values(by=["Symbol", "Date"])

arma_orders = {
    "TATAMOTORS": (2, 2),
    "TATASTEEL":  (2, 2),
    "TCS":        (1, 1),
}

results_arma = {}

for sym, (p, q) in arma_orders.items():
    print(f"\n=== {sym} — ARMA({p},{q}) on ΔClose ===")
    # Extract time series
    ts = df_diff[df_diff["Symbol"] == sym].copy()
    ts.set_index("Date", inplace=True)
    y = ts["Close_diff"].dropna()

    n = len(y)
    split = int(0.8 * n)
    train, test = y.iloc[:split], y.iloc[split:]

    model = ARIMA(train, order=(p, 0, q))
    fit = model.fit()

    try:
        print(f"\nARMA({p},{q}) model summary for {sym}:")
        print(fit.summary())

        print("\nCoefficient Estimates:")
        print("Params :", fit.params.values)
        print("StdErr :", fit.bse.values)
        print("95% CI:\n", fit.conf_int())

    except Exception as e:
        print(f"Failed to display summary for ARMA({p},{q}) — {sym}: {e}")

    lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
    Q = lb.loc[10, "lb_stat"]
    pval = lb.loc[10, "lb_pvalue"]
    print(f"Ljung–Box Q(10) = {Q:.2f}, p = {pval:.3f}")

    plt.figure(figsize=(6, 3))
    plot_acf(fit.resid, lags=20, ax=plt.gca(),
             title=f"{sym} ARMA({p},{q}) Residual ACF")
    plt.tight_layout()
    plt.show()

    pred = fit.get_prediction(start=split, end=n-1, dynamic=False)
    forecast = pred.predicted_mean
    conf_int = pred.conf_int()
    forecast.index = test.index
    conf_int.index = test.index

    mse = mean_squared_error(test, forecast)
    aic = fit.aic
    bic = fit.bic

    results_arma[sym] = dict(
        mse=mse, aic=aic, bic=bic,
        forecast=forecast,
        lower_ci=conf_int.iloc[:, 0],
        upper_ci=conf_int.iloc[:, 1],
        test=test, train=train,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(train, label="Train ΔClose", color="blue")
    plt.plot(test, label="Test ΔClose", color="black")
    plt.plot(forecast, label="Forecast", color="orange")
    plt.fill_between(test.index,
                     conf_int.iloc[:, 0],
                     conf_int.iloc[:, 1],
                     color="gray", alpha=0.4)
    plt.title(f"{sym} — ARMA({p},{q}) Forecast on ΔClose")
    plt.legend()
    plt.tight_layout()
    plt.show()

for sym, r in results_arma.items():
    print(f"{sym}: MSE={r['mse']:.2f}  AIC={r['aic']:.2f}  BIC={r['bic']:.2f}")

#%%
'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

results_arima = {}

for sym in df['Symbol'].unique():
    # 1) Extract and split
    g = (df[df['Symbol'] == sym]
           .sort_values('Date')
           .set_index('Date'))
    y = g['Close'].astype(float)
    n_train = int(0.8 * len(y))
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    best = {
        'p': None, 'd': 1, 'q': None,
        'mse': np.inf, 'aic': np.nan, 'bic': np.nan,
        'lb_stat': None, 'lb_pvalue': None
    }

    # 2) Grid‐search p=1..4, q=0..4
    for p in range(1, 5):
        for q in range(0, 5):
            try:
                # force no constant because d=1
                model = ARIMA(y_train, order=(p, 1, q), trend='n')
                fit = model.fit()

                # 2a) Ljung–Box test on residuals (lag=10)
                lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)
                lb_stat = lb['lb_stat'].iloc[0]
                lb_pvalue = lb['lb_pvalue'].iloc[0]
                # require whiteness: p-value > 0.05
                if lb_pvalue <= 0.05:
                    continue

                # 2b) one‐step forecast on the whole test set
                fc = fit.forecast(steps=len(y_test))
                mse = mean_squared_error(y_test, fc)

                # 2c) pick best by test‐MSE
                if mse < best['mse']:
                    best.update({
                        'p': p, 'q': q,
                        'mse': mse,
                        'aic': fit.aic,
                        'bic': fit.bic,
                        'lb_stat': lb_stat,
                        'lb_pvalue': lb_pvalue
                    })
            except Exception:
                continue

    # 3) Report
    if best['p'] is not None:
        print(f"{sym}  ARIMA best (p,d,q)=({best['p']},{best['d']},{best['q']})  "
              f"MSE={best['mse']:.2f}  AIC={best['aic']:.2f}  BIC={best['bic']:.2f}  "
              f"Ljung–Box(10) Q={best['lb_stat']:.2f}, p={best['lb_pvalue']:.3f}")
    else:
        print(f"{sym}  ✗ no ARIMA(·,1,·) model passed the Ljung–Box test.")
    results_arima[sym] = best

'''

#%%

df = pd.read_csv("filtered_stock_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
data = df.pivot(index='Date', columns='Symbol', values='Close').sort_index()

overfitted_orders = {
    "TATAMOTORS": (3, 1, 2),
    "TATASTEEL":  (2, 1, 2),
    "TCS":        (1, 1, 1),
}

results = {}

for sym, order in overfitted_orders.items():
    ts = data[sym].dropna()
    train_size = int(len(ts) * 0.8)
    train, test = ts.iloc[:train_size], ts.iloc[train_size:]

    best_model = None
    best_aic = np.inf
    best_result = None

    for p in range(order[0] + 1):
        for q in range(order[2] + 1):
            try:
                model = ARIMA(train, order=(p, order[1], q), trend="t")
                result = model.fit()

                if all(result.pvalues < 0.05) and result.aic < best_aic:
                    best_model = model
                    best_result = result
                    best_aic = result.aic
            except:
                continue

    if best_result is None:
        best_model = ARIMA(train, order=order, trend="t")
        best_result = best_model.fit()

    pred = best_result.get_prediction(start=train_size, end=train_size + len(test) - 1, dynamic=False)
    mean_forecast = pred.predicted_mean
    conf_int = pred.conf_int()
    mean_forecast.index = test.index
    conf_int.index = test.index

    mse = mean_squared_error(test, mean_forecast)
    rmse = sqrt(mse)
    aic = best_result.aic
    bic = best_result.bic
    lb = acorr_ljungbox(best_result.resid, lags=[10], return_df=True)
    Q = lb.loc[10, "lb_stat"]
    pval = lb.loc[10, "lb_pvalue"]

    results[sym] = dict(
        mse=mse, rmse=rmse, aic=aic, bic=bic,
        forecast=mean_forecast,
        lower_ci=conf_int.iloc[:, 0],
        upper_ci=conf_int.iloc[:, 1],
        test=test, train=train,
        result=best_result,
        Q=Q, pval=pval
    )

    plt.figure(figsize=(10, 4))
    plt.plot(train, label="Train")
    plt.plot(test, label="Test", color="gray")
    plt.plot(mean_forecast, label="Forecast", color="orange")
    plt.fill_between(mean_forecast.index,
                     conf_int.iloc[:, 0].astype(float),
                     conf_int.iloc[:, 1].astype(float),
                     color="lightgray", alpha=0.5)
    plt.title(f"{sym} — Final ARIMA Forecast")
    plt.legend()
    plt.tight_layout()
    plt.show()

for sym, res in results.items():
    print(f"{'='*22}  {sym}  {'='*22}")
    print("model summary for", sym + ":\n", res['result'].summary())
    print("Coefficient Estimates:")
    print("Params :", res['result'].params.values)
    print("StdErr :", res['result'].bse.values)
    print("95% CI:\n", res['result'].conf_int())
    print(f"In‐sample Ljung–Box Q(10) = {res['Q']:.2f}, p = {res['pval']:.3f}")
    print(f"MSE={res['mse']:.2f}  RMSE={res['rmse']:.2f}  AIC={res['aic']:.2f}  BIC={res['bic']:.2f}")
    print()

#%%

from scipy.linalg import toeplitz

df_diff = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])

raw = (
    pd.concat([
        pd.read_csv("TCS.csv",        parse_dates=["Date"]),
        pd.read_csv("TATAMOTORS.csv", parse_dates=["Date"]),
        pd.read_csv("TATASTEEL.csv",  parse_dates=["Date"])
    ], ignore_index=True)
    .loc[:, ["Date","Symbol","Volume","Turnover","Trades","Deliverable Volume","%Deliverble"]]
)

df = pd.merge(df_diff, raw, on=["Date","Symbol"], how="inner")

df = df.sort_values(["Symbol","Date"])
for col in ["Volume","Turnover","Trades","Deliverable Volume","%Deliverble"]:
    df[f"{col}_diff"] = df.groupby("Symbol")[col].diff()


df = df.replace([np.inf,-np.inf],np.nan).dropna()

diff_cols = [c for c in df.columns if c.endswith("_diff") and c!="Close_diff"]

best_inputs = {}
for sym, g in df.groupby("Symbol"):
    corrs = g[diff_cols].corrwith(g["Close_diff"])
    best = corrs.abs().idxmax()
    best_inputs[sym] = best
    print(f"{sym}: best input → {best}  (corr={corrs[best]:.3f})")

def gpac_table(acf_vals, max_k=7, max_j=7):
    gp = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k+1):
            D = np.array([[acf_vals[abs(j+i-m)] if abs(j+i-m)<len(acf_vals) else 0
                            for m in range(k)] for i in range(k)])
            N = D.copy()
            for i in range(k):
                lag = j + i + 1
                N[i, -1] = acf_vals[lag] if lag<len(acf_vals) else 0
            dD, dN = np.linalg.det(D), np.linalg.det(N)
            gp[j, k-1] = np.nan if np.isclose(dD,0) else dN/dD
    idx = [f"j={j}" for j in range(max_j)]
    cols = [f"k={k}" for k in range(1,max_k+1)]
    return pd.DataFrame(gp, index=idx, columns=cols).round(3).replace(-0.0,0.0)

from statsmodels.tsa.stattools import acf

for sym, g in df.groupby("Symbol"):
    u = g[best_inputs[sym]].values
    y = g["Close_diff"].values
    N = len(u); K = 7

    Ru  = np.array([np.dot(u[:N-t], u[t:])/(N-t) for t in range(K+1)])
    Ruy = np.array([np.dot(u[:N-t], y[t:])/(N-t) for t in range(K+1)])
    g_hat = np.linalg.solve(toeplitz(Ru[:K]), Ruy[1:K+1])

    acf_g = acf(g_hat, nlags=20, fft=False)
    gpac_g = gpac_table(acf_g)

    acf_y = acf(y, nlags=20, fft=False)
    gpac_h = gpac_table(acf_y)

    fig, ax = plt.subplots(1,2,figsize=(12,5))
    sns.heatmap(gpac_g, annot=True, cmap="RdBu_r", vmin=-2, vmax=2, ax=ax[0])
    ax[0].set_title(f"{sym} — G-GPAC")
    sns.heatmap(gpac_h, annot=True, cmap="RdBu_r", vmin=-2, vmax=2, ax=ax[1])
    ax[1].set_title(f"{sym} — H-GPAC")
    plt.tight_layout()
    plt.show()

#%%

def compute_error_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    y_g = np.zeros(N)
    e = np.zeros(N)

    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -np.array(theta[nb - 1 + nf + nc:])]

    max_lag = max(len(b), len(f))

    for t in range(max_lag, N):
        y_g[t] = sum(b[i] * u[t - i] for i in range(len(b))) - sum(f[j] * y_g[t - j] for j in range(1, len(f)))

    residual = y - y_g

    max_h_lag = max(len(c), len(d))
    for t in range(max_h_lag, N):
        num = sum(d[j] * residual[t - j] for j in range(len(d)))
        den = sum(c[i] * e[t - i] for i in range(1, len(c)))
        e[t] = num - den

    return e[max_h_lag:]

def compute_jacobian_bj(theta, y, u, nb, nf, nc, nd, delta=1e-7):
    base_error = compute_error_bj(theta, y, u, nb, nf, nc, nd)
    X = np.zeros((len(base_error), len(theta)))

    for i in range(len(theta)):
        perturbed = theta.copy()
        perturbed[i] += delta
        perturbed_error = compute_error_bj(perturbed, y, u, nb, nf, nc, nd)
        X[:, i] = (base_error - perturbed_error) / delta

    return X

def levenberg_marquardt_bj(y, u, theta_init, nb, nf, nc, nd, mu_init=0.01, max_iter=100, epsilon=1e-3):
    theta = theta_init.copy()
    mu = mu_init
    sse_track = []

    for it in range(max_iter):
        e = compute_error_bj(theta, y, u, nb, nf, nc, nd)
        SSE = e @ e
        sse_track.append(SSE)

        X = compute_jacobian_bj(theta, y, u, nb, nf, nc, nd)
        A = X.T @ X
        g = X.T @ e

        try:
            delta_theta = np.linalg.inv(A + mu * np.eye(len(theta))) @ g
        except LinAlgError:
            print(f"Warning: Singular matrix at iteration {it}, using pseudoinverse instead.")
            delta_theta = pinv(A + mu * np.eye(len(theta))) @ g

        theta_new = theta + delta_theta
        e_new = compute_error_bj(theta_new, y, u, nb, nf, nc, nd)
        SSE_new = e_new @ e_new

        print(f"Iter {it} | SSE: {SSE:.4f} | mu: {mu:.2e} | Δθ norm: {np.linalg.norm(delta_theta):.2e}")

        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:
                print("Converged.")
                break
            theta = theta_new
            mu /= 10
        else:
            mu *= 10

    return theta, sse_track

def confidence_intervals_bj(theta_est, y, u, nb, nf, nc, nd):
    N = len(y)
    p = len(theta_est)
    e = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)
    sse = np.sum(e ** 2)
    sigma2 = sse / (N - p)

    X = compute_jacobian_bj(theta_est, y, u, nb, nf, nc, nd)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    stderr = np.sqrt(np.diag(cov))

    tval = t_dist.ppf(0.975, N - p)
    lower = theta_est - tval * stderr
    upper = theta_est + tval * stderr

    return list(zip(lower, theta_est, upper))

#%%

def is_significant(ci):
    return all(lb * ub > 0 for lb, _, ub in ci)

def calculate_aic_bic(e, num_params):
    N = len(e)
    sse = np.sum(e ** 2)
    sigma2 = sse / N
    log_likelihood = -0.5 * N * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    aic = -2 * log_likelihood + 2 * num_params
    bic = -2 * log_likelihood + num_params * np.log(N)
    return aic, bic

def q_test(e, K=50, nc=1, nd=1, alpha=0.05):
    N = len(e)
    e = e - np.mean(e)
    e = e / np.std(e)
    Q = 0
    for tau in range(1, K + 1):
        r_tau = np.sum(e[tau:] * e[:-tau]) / (N - tau)
        Q += r_tau ** 2
    Q_stat = N * Q
    dof = K - nc - nd
    Q_crit = chi2.ppf(1 - alpha, df=dof)
    return Q_stat, Q_crit, dof

def s_test(e, u, theta_est, nb, nf, K=20, significance=0.05):
    N = len(e)
    e = e - np.mean(e)
    e = e / np.std(e)

    f = np.r_[1, theta_est[nb - 1: nb - 1 + nf]]
    alpha_t = np.zeros_like(u)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t - j] for j in range(1, len(f)))
    alpha_t = alpha_t - np.mean(alpha_t)
    alpha_t = alpha_t / np.std(alpha_t)

    S = 0.0
    for tau in range(K + 1):
        r_ae = np.sum(alpha_t[:N - tau] * e[tau:]) / (N - tau)
        S += r_ae ** 2

    S_stat = float(N * S)
    dof = K - (nb - 1) - nf
    S_crit = float(chi2.ppf(1 - significance, df=dof))
    return S_stat, S_crit, dof

#%%

np.seterr(over='ignore', invalid='ignore')

best_inputs = {
    "TATAMOTORS": "Turnover_diff",
    "TATASTEEL": "Volume_diff",
    "TCS": "%Deliverble_diff"
}

fixed_orders = {
    "TATAMOTORS": (4, 3, 3, 3),
    "TATASTEEL": (4, 3, 3, 3),
    "TCS": (2, 2, 2, 2)
}

def evaluate_model_relaxed(y, u, nb, nf, nc, nd):
    p = (nb - 1) + nf + nc + nd
    theta_init = np.zeros(p)
    theta_est, sse_track = levenberg_marquardt_bj(y, u, theta_init, nb, nf, nc, nd)
    ci = confidence_intervals_bj(theta_est, y, u, nb, nf, nc, nd)

    b_ci = ci[: (nb - 1)]
    if (nb - 1) > 0 and not all(lb * ub > 0 for lb, _, ub in b_ci):
        return None

    residuals = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)

    N = len(residuals)
    sse = np.sum(residuals ** 2)
    sigma2 = sse / N
    loglike = -0.5 * N * (np.log(2 * np.pi) + np.log(sigma2) + 1)
    aic = -2 * loglike + 2 * p
    bic = -2 * loglike + p * np.log(N)

    q_stat, q_crit, q_dof = q_test(residuals, nc=nc, nd=nd)
    s_stat, s_crit, s_dof = s_test(residuals, u, theta_est, nb, nf)

    return {
        'order': (nb, nf, nc, nd),
        'theta': theta_est,
        'ci': ci,
        'aic': aic, 'bic': bic,
        'q_stat': q_stat, 'q_crit': q_crit, 'q_dof': q_dof,
        's_stat': s_stat, 's_crit': s_crit, 's_dof': s_dof
    }

results = {}
for sym in df['Symbol'].unique():
    print(f"\n>>> Evaluating fixed model for {sym}")
    g = df[df['Symbol'] == sym]

    y = (g['Close_diff'].values - g['Close_diff'].mean()) / g['Close_diff'].std()
    ucol = best_inputs[sym]
    u = (g[ucol].values - g[ucol].mean()) / g[ucol].std()

    nb, nf, nc, nd = fixed_orders[sym]
    best = evaluate_model_relaxed(y, u, nb, nf, nc, nd)

    if not best:
        print("  ✗ model estimation failed or b coefficients not significant")
        continue

    o = best['order']
    print(f"✔ Best   order={o}   AIC={best['aic']:.2f}   BIC={best['bic']:.2f}")

    print("  Input‐b coefficients (with 95% CI):")
    for i in range(o[0] - 1):
        lb, est, ub = best['ci'][i]
        print(f"    b[{i + 1}] = {est:.4f}   CI=[{lb:.4f},{ub:.4f}]")

    print("  All θ coefficients (with 95% CI):")
    for idx, (lb, est, ub) in enumerate(best['ci']):
        print(f"    θ[{idx + 1}] = {est:.4f}   CI=[{lb:.4f},{ub:.4f}]")

    print(f"  Ljung–Box Q={best['q_stat']:.2f} (crit={best['q_crit']:.2f}, df={best['q_dof']})")
    print(f"  X–res cross‐corr S={best['s_stat']:.2f} (crit={best['s_crit']:.2f}, df={best['s_dof']})")

    results[sym] = best

#%%

final_orders = {
    'TCS':        (2, 2, 2, 2),
    'TATAMOTORS': (4, 3, 3, 3),
    'TATASTEEL':  (4, 3, 3, 3),
}

for sym, (nb, nf, nc, nd) in final_orders.items():
    print(f"\n--- Residual diagnostics for {sym} ---")

    g = df[df['Symbol'] == sym].sort_values("Date")
    y0 = g['Close_diff'].values
    u0 = g[best_inputs[sym]].values
    y = (y0 - y0.mean()) / y0.std()
    u = (u0 - u0.mean()) / u0.std()

    p = (nb - 1) + nf + nc + nd
    theta_init = np.zeros(p)
    theta_est, _ = levenberg_marquardt_bj(y, u, theta_init, nb, nf, nc, nd)

    resid = compute_error_bj(theta_est, y, u, nb, nf, nc, nd)

    y_std = y0.std()
    rmse_val = np.sqrt(mean_squared_error(y0[-len(resid):], y0[-len(resid):] - resid * y_std))
    print(f"RMSE: {rmse_val:.2f}")

    lb = acorr_ljungbox(resid, lags=[50], return_df=True)
    Q, pQ = lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0]
    print(f"Ljung–Box (lag=10): Q = {Q:.2f}, p = {pQ:.3f}")

    S, S_crit, S_df = s_test(resid, u, theta_est, nb, nf, K=50)
    print(f"S-test: S = {S:.2f}, crit = {S_crit:.2f}, df = {S_df}")

    plt.figure(figsize=(6,3))
    plot_acf(resid, lags=20, zero=False)
    plt.title(f"{sym} — Residual ACF")
    plt.tight_layout()
    plt.show()

#%%

final_orders = {
    'TATAMOTORS': (4, 3, 3, 3),
    'TATASTEEL':  (4, 3, 3, 3),
    'TCS':        (2, 2, 2, 2)
}

np.seterr(all='ignore')

for sym, (nb, nf, nc, nd) in final_orders.items():
    print(f"\n=== Residual Diagnostics for {sym} ===")

    theta = results[sym]['theta']

    g = df[df['Symbol']==sym]
    y = (g['Close_diff'].values - g['Close_diff'].mean()) / g['Close_diff'].std()
    ucol = best_inputs[sym]
    u = (g[ucol].values - g[ucol].mean()) / g[ucol].std()

    e = compute_error_bj(theta, y, u, nb, nf, nc, nd)
    N = len(e)

    resid_var = np.var(e, ddof=1)
    print(f"Residual variance σ² = {resid_var:.4f}")

    X = compute_jacobian_bj(theta, y, u, nb, nf, nc, nd)
    sigma2_hat = np.sum(e**2)/(N - len(theta))
    cov_theta = sigma2_hat * inv(X.T @ X)
    print("Covariance matrix of θ (rounded):")
    print(np.round(cov_theta,4))

    bias = np.mean(e)
    print(f"Mean(residuals) = {bias:.4e} → {'biased' if abs(bias)>1e-2 else '≈ unbiased'}")

    print(f"Forecast‐error variance ≈ {resid_var:.4f}")

    b = np.r_[1, theta[:nb-1]]
    f = np.r_[1, theta[nb-1:nb-1+nf]]
    c = np.r_[1, theta[nb-1+nf:nb-1+nf+nc]]

    roots_f = np.roots(f)
    roots_c = np.roots(c)
    tol = 1e-2
    common = [rf for rf in roots_f for rc in roots_c if abs(rf-rc)<tol]
    if common:
        new_f = np.real_if_close(np.poly([r for r in roots_f if all(abs(r-r0)>tol for r0 in common)]), tol)
        new_c = np.real_if_close(np.poly([r for r in roots_c if all(abs(r-r0)>tol for r0 in common)]), tol)
        print("Zero–pole cancellation found, simplified polynomials:")
        print(" F_simplified(q):", np.round(new_f,4))
        print(" C_simplified(q):", np.round(new_c,4))
    else:
        print("No zero–pole cancellation between F and C.")

#%%

df_diff = pd.read_csv("filtered_stock_data.csv", parse_dates=["Date"])
raw = pd.concat([
    pd.read_csv("TCS.csv",        parse_dates=["Date"]),
    pd.read_csv("TATAMOTORS.csv", parse_dates=["Date"]),
    pd.read_csv("TATASTEEL.csv",  parse_dates=["Date"])
], ignore_index=True)[[
    "Date","Symbol","Volume","Turnover","Trades","Deliverable Volume","%Deliverble"
]]
df = pd.merge(df_diff, raw, on=["Date","Symbol"], how="inner")
df.sort_values(["Symbol","Date"], inplace=True)
for col in ["Volume","Turnover","Trades","Deliverable Volume","%Deliverble"]:
    df[f"{col}_diff"] = df.groupby("Symbol")[col].diff()
df.replace([np.inf,-np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

best_inputs = {
    "TATAMOTORS": "Turnover_diff",
    "TATASTEEL":  "Volume_diff",
    "TCS":        "%Deliverble_diff"
}
fixed_orders = {
    "TATAMOTORS": (4, 3, 3, 3),
    "TATASTEEL":  (4, 3, 3, 3),
    "TCS":        (2, 2, 2, 2)
}

def plot_one_step(y, e_full, phi, psi, n_plot=30):

    dfp = pd.DataFrame({"y": y, "e": e_full})
    for i in range(1, len(phi)):
        dfp[f"y_lag{i}"] = dfp["y"].shift(i)
    for j in range(1, len(psi)):
        dfp[f"e_lag{j}"] = dfp["e"].shift(j)


    dfp["y_hat"] = 0.0
    for i in range(1, len(phi)):
        dfp["y_hat"] += phi[i] * dfp[f"y_lag{i}"]
    for j in range(1, len(psi)):
        dfp["y_hat"] += psi[j] * dfp[f"e_lag{j}"]

    dfp.dropna(inplace=True)
    Np = min(n_plot, len(dfp))
    plt.figure(figsize=(10,4))
    plt.plot(dfp["y"].iloc[-Np:],       label="Actual")
    plt.plot(dfp["y_hat"].iloc[-Np:], "--", label="1-step Forecast")
    plt.title("1-Step Ahead Forecast (Box–Jenkins)")
    plt.xlabel("Time Index"); plt.ylabel("Normalized Close_diff")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

def plot_h_step(theta, y_hist, u_hist, nb, nf, nc, nd, h=20):
    b = np.r_[1, theta[:nb-1]]
    f = np.r_[1, theta[nb-1:nb-1+nf]]
    c = np.r_[1, theta[nb-1+nf:nb-1+nf+nc]]
    d = np.r_[1, -theta[nb-1+nf+nc:]]

    yhat = np.zeros(h)
    ehat = np.zeros(h)
    seed = min(nf, len(y_hist))
    yhat[:seed] = y_hist[-seed:]

    for t in range(seed, h):
        ar = sum(b[i] * u_hist[t-i]    for i in range(len(b))    if t-i>=0)
        tr = sum(f[j] * yhat[t-j]      for j in range(1,len(f)) if t-j>=0)
        me = sum(d[j] * ehat[t-j]      for j in range(len(d))    if t-j>=0)
        mc = sum(c[i] * ehat[t-i]      for i in range(1,len(c)) if t-i>=0)
        yhat[t] = ar - tr + me - mc

    plt.figure(figsize=(10,4))
    plt.plot(range(len(y_hist)), y_hist, label="History")
    plt.plot(range(len(y_hist), len(y_hist)+h),
             yhat[:h], "--", label=f"{h}-Step Forecast")
    plt.title(f"{h}-Step Ahead Forecast (Box–Jenkins)")
    plt.xlabel("Time Index"); plt.ylabel("Normalized Close_diff")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

for sym in ["TATAMOTORS","TATASTEEL","TCS"]:
    print(f"\n=== {sym} ===")
    data = df[df["Symbol"]==sym].dropna(
        subset=["Close_diff", best_inputs[sym]]
    )

    y = (data["Close_diff"] - data["Close_diff"].mean()) / data["Close_diff"].std()
    u = (data[best_inputs[sym]] - data[best_inputs[sym]].mean()) / data[best_inputs[sym]].std()
    nb,nf,nc,nd = fixed_orders[sym]

    theta0 = np.zeros((nb-1)+nf+nc+nd)
    theta, _ = levenberg_marquardt_bj(y.values, u.values, theta0, nb,nf,nc,nd)

    res = compute_error_bj(theta, y.values, u.values, nb,nf,nc,nd)

    lag = max(nc+1, nd+1)
    e_full = np.full(len(y), np.nan)
    e_full[lag:] = res

    phi = np.r_[1, theta[nb-1:nb-1+nf]]
    psi = np.r_[1, theta[nb-1+nf:nb-1+nf+nc]]

    plot_one_step(y.values, e_full, phi, psi, n_plot=30)
    plot_h_step(theta, y.values, u.values, nb,nf,nc,nd, h=20)

#%%

h = 30

final_models = {
    "TATAMOTORS": "ARMA",
    "TATASTEEL": "BJ"
}

final_orders = {
    'TATAMOTORS': (2, 0, 2),
    'TATASTEEL': (4, 3, 3, 3)
}

for sym in final_models:
    print(f"\nForecast for {sym}")

    # Prepare data
    g = df[df['Symbol'] == sym].sort_values("Date")
    y_all = g['Close_diff'].values
    u_all = g[best_inputs[sym]].values

    split = int(len(y_all) * 0.8)
    y_train, y_test = y_all[:split], y_all[split:]
    u_train, u_test = u_all[:split], u_all[split:]

    if final_models[sym] == "ARMA":
        p, d, q = final_orders[sym]
        model = ARIMA(y_train, order=(p, d, q)).fit()
        fc = model.forecast(steps=len(y_test))
    else:
        nb, nf, nc, nd = final_orders[sym]


        y_mean, y_std = y_train.mean(), y_train.std()
        u_mean, u_std = u_train.mean(), u_train.std()
        y_norm = (y_train - y_mean) / y_std
        u_norm = (u_train - u_mean) / u_std

        theta_init = np.zeros((nb - 1) + nf + nc + nd)
        theta_est, _ = levenberg_marquardt_bj(y_norm, u_norm, theta_init, nb, nf, nc, nd)

        u_future = (u_test - u_mean) / u_std
        y_init = y_norm[-max(nf, nb):]

        def multi_step_forecast_bj(theta, y_init, u_future, nb, nf, nc, nd):
            steps = len(u_future)
            b = np.r_[1, theta[:nb - 1]]
            f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
            c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
            d = np.r_[1, -np.array(theta[nb - 1 + nf + nc:])]

            yhat = list(y_init.copy())
            ehat = [0] * max(len(c), len(d))

            for t in range(steps):
                u_terms = sum(b[i] * u_future[t - i] if t - i >= 0 else 0 for i in range(len(b)))
                y_terms = sum(f[j] * yhat[-j] for j in range(1, len(f)))
                num = sum(d[j] * (yhat[-j] - u_terms + y_terms) for j in range(len(d)))
                den = sum(c[i] * ehat[-i] for i in range(1, len(c)))
                e_new = num - den
                y_new = u_terms - y_terms + e_new
                yhat.append(y_new)
                ehat.append(e_new)

            return np.array(yhat[-steps:]) * y_std + y_mean  # denormalize

        fc = multi_step_forecast_bj(theta_est, y_init, u_future, nb, nf, nc, nd)

    rmse = sqrt(mean_squared_error(y_test[:len(fc)], fc))
    print(f"RMSE = {rmse:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(y_train)), y_train, label="Train")
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label="Test", color='tab:orange')
    plt.plot(range(len(y_train), len(y_train) + len(fc)), fc, '--', label="Forecast", color='tab:green')
    plt.title(f"{sym} — {final_models[sym]} Forecast vs Actual")
    plt.xlabel("Time Index")
    plt.ylabel("Close_diff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#%%

#%%

def h_step_forecast_bj(theta, y_hist, u_future, nb, nf, nc, nd, h):
    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -theta[nb - 1 + nf + nc:]]

    yhat = list(y_hist[-nf:])
    ehat = [0] * h
    pad_len = len(b) - 1
    u_padded = np.r_[np.zeros(pad_len), u_future]

    forecast = []
    for t in range(h):
        ar = sum(f[j] * yhat[-j] for j in range(1, len(f)))
        ma = sum(c[j] * ehat[t - j] for j in range(1, min(len(c), t + 1)))
        inp = sum(b[i] * u_padded[t + pad_len - i] for i in range(len(b)))

        y_pred = inp - ar + ma
        forecast.append(y_pred)
        yhat.append(y_pred)

    return np.array(forecast)

final_orders = {
    'TCS': (2, 2, 2, 2)
}
best_inputs = {
    'TCS': '%Deliverble_diff'
}

for sym, (nb, nf, nc, nd) in final_orders.items():
    print(f"\n--- h-step Forecast for {sym} ---")

    g = df[df['Symbol'] == sym].sort_values("Date")
    y0 = g['Close_diff'].values
    u0 = g[best_inputs[sym]].values

    y = (y0 - y0.mean()) / y0.std()
    u = (u0 - u0.mean()) / u0.std()

    split = int(len(y) * 0.8)
    y_train, y_test = y[:split], y[split:]
    u_train, u_test = u[:split], u[split:]

    p = (nb - 1) + nf + nc + nd
    theta_init = np.zeros(p)
    theta_est, _ = levenberg_marquardt_bj(y_train, u_train, theta_init, nb, nf, nc, nd)

    h = len(y_test)
    y_pred = h_step_forecast_bj(theta_est, y_train, u_test, nb, nf, nc, nd, h)

    y_test_orig = y0[split:]
    y_pred_orig = y_pred * y0.std() + y0.mean()

    rmse = sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    print(f"{sym} Forecast RMSE = {rmse:.2f}")

    plt.figure(figsize=(10, 4))
    plt.plot(g['Date'].values[split:], y_test_orig, label='Actual')
    plt.plot(g['Date'].values[split:], y_pred_orig, label='Forecast', linestyle='--')
    plt.title(f"{sym} — h-step Forecast (Box–Jenkins)\nRMSE = {rmse:.2f}")
    plt.xlabel("Date")
    plt.ylabel("ΔClose Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
