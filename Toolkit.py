import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from scipy.linalg import toeplitz
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor

def apply_differencing(data, lag):
    if len(data) <= lag:
        raise ValueError("Data length must be greater than the differencing lag.")
    return data[lag:] - data[:-lag]

def apply_combined_differencing(df, column_name='Temp',
                                     order=1, seasonality=0,
                                     new_column=True):

    if order < 0 or seasonality < 0:
        raise ValueError("Both order and seasonality must be >= 0.")

    diff_series = df[column_name].copy()

    if seasonality > 0:
        diff_series = diff_series.diff(periods=seasonality)

    for _ in range(order):
        diff_series = diff_series.diff()

    diff_series = diff_series.dropna()

    if new_column:
        suffix = f"_diff{order}" if order > 0 else ""
        suffix += f"_seasonal{seasonality}" if seasonality > 0 else ""
        new_col_name = f"{column_name}{suffix}"
        df[new_col_name] = pd.Series(diff_series.values, index=diff_series.index)
    else:
        df = df.iloc[len(df) - len(diff_series):].copy()
        df[column_name] = diff_series.values

    return df

def apply_log_differencing(df, column_name='Temp', lag=1, new_column=True):

    log_series = np.log(df[column_name])
    log_diff_series = log_series.diff(periods=lag).dropna()

    if new_column:
        diff_col_name = f"log_diff_{column_name}_lag{lag}"
        df[diff_col_name] = pd.Series(log_diff_series.values, index=log_diff_series.index)
    else:
        df = df.iloc[lag:].copy()
        df[column_name] = log_diff_series.values

    return df

# Rolling means and variance

def Cal_rolling_mean_var(data, column_name):

    rolling_mean = []
    rolling_variance = []

    for i in range(1, len(data) + 1):

        subset = data[column_name].iloc[:i]


        mean = subset.mean()
        variance = subset.var(ddof=0)


        rolling_mean.append(mean)
        rolling_variance.append(variance)


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))


    ax[0].plot(range(1, len(data) + 1), rolling_mean, label=f'Rolling Mean of {column_name}')
    ax[0].set_title(f'Rolling Mean of {column_name}')
    ax[0].set_xlabel('Number of Samples')
    ax[0].set_ylabel('Mean')
    ax[0].legend()




    ax[1].plot(range(1, len(data) + 1), rolling_variance, label=f'Rolling Variance of {column_name}', color='red')
    ax[1].set_title(f'Rolling Variance of {column_name}')
    ax[1].set_xlabel('Number of Samples')
    ax[1].set_ylabel('Variance')
    ax[1].legend()


    plt.tight_layout()
    plt.show()


# ADF test

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# KPSS test

def kpss_test(timeseries):
    print('Results of KPSS Test:')

    kpsstest = kpss(timeseries, regression='c', nlags="auto")


    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])


    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value

    print(kpss_output)

# ACF

def acf_man(x, lag):
    n = len(x)
    x_mean = np.mean(x)

    if lag >= 0:
        numerator = np.sum((x[:n - lag] - x_mean) * (x[lag:] - x_mean))
    else:
        numerator = np.sum((x[-lag:] - x_mean) * (x[:n + lag] - x_mean))

    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator


def plot_acf_manual(data, lags=20, title="Autocorrelation Function"):

    acf_values = [acf_man(data, lag) for lag in range(-lags, lags + 1)]

    plt.figure(figsize=(8, 5))
    plt.stem(range(-lags, lags + 1), acf_values, linefmt='b-', markerfmt='ro', basefmt='k')
    plt.axhspan(-1.96 / np.sqrt(len(data)), 1.96 / np.sqrt(len(data)), alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return acf_values

# VIF

def calculate_vif(X):

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# GPAC

def gpac_table(acf, max_k: 7, max_j: 7, model_nb: int = None):
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
                        D[i, m] = acf[lag] if lag < len(acf) else 0

                N = D.copy()
                for i in range(k):
                    lag = j + i + 1
                    N[i, -1] = acf[lag] if lag < len(acf) else 0

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

    df = pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)], columns=[f"k={k+1}" for k in range(max_k)])
    df = df.round(3)
    df = df.replace(-0.0, 0.0)
    df = df.applymap(lambda x: 0.0 if np.isclose(x, 0, atol=1e-3) else x)
    return df

def plot_gpac_table(gpac_df):
    plt.figure(figsize=(10, 8))
    sns.heatmap(gpac_df, annot=True, fmt=".3f", cmap="rocket_r", cbar=True, vmin=-2, vmax=2)
    plt.title("Generalized Partial Autocorrelation (GPAC) Table", fontsize=14)
    plt.xlabel("k (AR order)")
    plt.ylabel("j (MA order)")
    plt.tight_layout()
    plt.show()

# ACF/PACF


def plot_acf_pacf(y, lags=20):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    sym_acf = np.concatenate((acf[:0:-1], acf))
    sym_pacf = np.concatenate((pacf[:0:-1], pacf))
    lag_size = np.arange(-lags, lags + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].stem(lag_size, sym_acf)
    axes[0].set_title('ACF Plot')
    axes[0].set_xlabel('Lags')
    axes[0].set_ylabel('ACF Values')
    axes[0].axhspan(-1.96 / np.sqrt(len(y)), 1.96 / np.sqrt(len(y)), alpha=0.2, color='blue')

    axes[1].stem(lag_size, sym_pacf)
    axes[1].set_title('PACF Plot')
    axes[1].set_xlabel('Lags')
    axes[1].set_ylabel('PACF Values')
    axes[1].axhspan(-1.96 / np.sqrt(len(y)), 1.96 / np.sqrt(len(y)), alpha=0.2, color='blue')

    plt.tight_layout(pad=3)
    plt.show()

# LM Estimation

def compute_error(theta, y, na, nb):
    N = len(y)
    e = np.zeros(N)

    for t in range(max(na, nb), N):
        ar_sum = sum([-theta[i] * y[t - i - 1] for i in range(na)]) if na > 0 else 0
        ma_sum = sum([theta[na + j] * e[t - j - 1] for j in range(nb)]) if nb > 0 else 0
        e[t] = y[t] + ar_sum - ma_sum

    return e[max(na, nb):]


def compute_jacobian(theta, y, na, nb, delta=1e-7):
    n_params = na + nb
    N = len(y)
    X = np.zeros((N - max(na, nb), n_params))

    base_error = compute_error(theta, y, na, nb)

    for i in range(n_params):
        theta_perturbed = theta.copy()
        theta_perturbed[i] += delta
        perturbed_error = compute_error(theta_perturbed, y, na, nb)
        X[:, i] = (base_error - perturbed_error) / delta

    return X


def levenberg_marquardt(y, theta_init, na, nb, mu_init=0.01, max_iter=100, epsilon=1e-3, mu_max=1e10):
    theta = theta_init.copy()
    mu = mu_init
    sse_track = []

    for iteration in range(max_iter):
        e = compute_error(theta, y, na, nb)
        SSE = e.T @ e
        sse_track.append(SSE)

        X = compute_jacobian(theta, y, na, nb)
        A = X.T @ X
        g = X.T @ e

        delta_theta = np.linalg.inv(A + mu * np.eye(len(theta))) @ g
        theta_new = theta + delta_theta
        e_new = compute_error(theta_new, y, na, nb)
        SSE_new = e_new.T @ e_new
        print(f"Iter {iteration} | SSE: {SSE:.4f} | mu: {mu:.4e} | theta: {theta} | delta norm: {np.linalg.norm(delta_theta):.4e}")

        if SSE_new < SSE:
            if np.linalg.norm(delta_theta) < epsilon:
                print(f"Converged in {iteration} iterations.")
                break
            theta = theta_new
            mu /= 10
        else:
            mu *= 10
            if mu > mu_max:
                print("Mu exceeded max limit. Stopping.")
                break


    return theta, sse_track




def confidence_intervals(theta_est, y, na, nb):

    N = len(y)
    p = na + nb
    e = compute_error(theta_est, y, na, nb)
    sse = np.sum(e ** 2)
    sigma2 = sse / (N - p)

    X = compute_jacobian(theta_est, y, na, nb)
    cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
    standard_errors = np.sqrt(np.diag(cov_matrix))

    lower_bounds = theta_est - 1.96 * standard_errors
    upper_bounds = theta_est + 1.96 * standard_errors

    print("\n95% Confidence Intervals:")
    for i, (lb, theta, ub) in enumerate(zip(lower_bounds, theta_est, upper_bounds)):
        param_name = f"a{i + 1}"
        print(f"{lb:.4f} < {param_name} < {ub:.4f}")




def covariance_matrix(theta_est, y, na, nb):

    N = len(y)
    p = na + nb
    e = compute_error(theta_est, y, na, nb)
    sse = np.sum(e ** 2)
    sigma2 = sse / (N - p)

    X = compute_jacobian(theta_est, y, na, nb)
    cov_matrix = sigma2 * np.linalg.inv(X.T @ X)

    print("\nEstimated Covariance Matrix of the Parameters:\n")
    print(cov_matrix)
    return cov_matrix

# zeroes/poles

def poles_zeros(ar_coeffs, ma_coeffs):

    ar_poly = np.r_[1, -np.array(ar_coeffs)] if ar_coeffs else np.array([1])
    ma_poly = np.r_[1, np.array(ma_coeffs)] if ma_coeffs else np.array([1])

    zeros = np.roots(ma_poly)
    poles = np.roots(ar_poly)

    print("\nPoles:")
    print(poles)

    print("\nZeros:")
    print(zeros)

    return poles, zeros

# Estimation with statsmodel

def estimate_with_statsmodels(y, ar_coeffs, ma_coeffs, na, nb):

    model = ARIMA(y, order=(na, 0, nb), trend='n')
    fit = model.fit()

    print("\n=== Model Summary ===")
    print(fit.summary())

    param_names = fit.param_names
    relevant_indices = [i for i, name in enumerate(param_names) if 'ar.L' in name or 'ma.L' in name]
    relevant_names = [param_names[i] for i in relevant_indices]
    relevant_params = fit.params[relevant_indices]
    ci = fit.conf_int()
    relevant_cis = ci[relevant_indices]



    print("\nEstimated Coefficients (AR/MA only, 3-digit precision):")
    for name, value in zip(np.array(param_names)[relevant_indices], relevant_params):
        print(f"{name}: {value:.3f}")

    print("\n=== 95% Confidence Intervals ===")
    for name, (ci_low, ci_high) in zip(relevant_names, relevant_cis):
        print(f"{ci_low:.4f} < {name} < {ci_high:.4f}")


# Q-test

def q_test(residuals, lags=50, model_df=0, alpha=0.05):

    residuals = np.asarray(residuals)
    N = len(residuals)
    residuals -= np.mean(residuals)
    var_e = np.var(residuals)

    Q = 0
    for tau in range(1, lags + 1):
        autocov = np.sum(residuals[tau:] * residuals[:-tau]) / (N - tau)
        r_tau = autocov / var_e
        Q += r_tau ** 2

    Q_stat = N * Q
    dof = lags - model_df
    Q_crit = chi2.ppf(1 - alpha, df=dof)

    print(f"\n--- Q-Test Summary ---")
    print(f"Q-statistic              : {Q_stat:.4f}")
    print(f"Chi-square Critical (α={alpha}, dof={dof}) : {Q_crit:.4f}")
    print("Result                   :",
          "✅ Residuals are white (Q < Q*)" if Q_stat < Q_crit
          else "❌ Residuals show autocorrelation (Q > Q*)")

    return Q_stat, Q_crit, dof

# s-test

def s_test(e, u, theta_est, nb, nf, K=20, significance=0.05):
    N = len(e)
    e = e - np.mean(e)

    f = np.r_[1, theta_est[nb - 1: nb - 1 + nf]]

    alpha_t = np.zeros_like(u)
    for t in range(len(f), N):
        alpha_t[t] = u[t] - sum(f[j] * alpha_t[t - j] for j in range(1, len(f)))

    alpha_t = alpha_t - np.mean(alpha_t)

    sigma_e = float(np.std(e))
    sigma_a = float(np.std(alpha_t))

    S = 0.0
    r_vals = []
    for tau in range(K + 1):
        R_ae = np.sum(alpha_t[:N - tau] * e[tau:]) / (N - tau)
        r_ae = R_ae / (sigma_a * sigma_e)
        r_vals.append(r_ae)
        S += r_ae ** 2

    S_stat = float(N * S)
    dof = K - (nb - 1) - nf
    S_crit = float(chi2.ppf(1 - significance, df=dof))

    print(f"S-stat: {S_stat:.4f}")
    print(f"Chi-square S* (α={significance}, DOF={dof}): {S_crit:.4f}")
    if S_stat < S_crit:
        print("G(q) is accurate (S < S*)")
    else:
        print("G(q) may be misspecified (S > S*)")

    return S_stat, S_crit, dof, r_vals


# ARMA FORECASTING

def forecast_arma(y, phi, theta, residuals, steps=1):

    p = len(phi)
    q = len(theta)
    y_hat = []
    y_hist = list(y)
    e_hist = list(residuals)

    for h in range(steps):
        ar_part = sum(phi[i] * y_hist[-i - 1] for i in range(p))
        ma_part = sum(theta[j] * (e_hist[-j - 1] if h == 0 else 0) for j in range(q))
        y_next = ar_part + ma_part
        y_hat.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)

    return y_hat


# ARMA

def one_step_forecast_arma_plot(y, e, phi, theta, n_plot=20):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)
    N = len(y)
    lag = max(p, q)

    # compute one-step forecasts
    y_hat = np.zeros(N)
    for t in range(lag, N):
        ar = sum(phi[i] * y[t-i-1]   for i in range(p))
        ma = sum(theta[j] * e[t-j-1] for j in range(q))
        y_hat[t] = ar + ma

    actual   = y[lag:]
    forecast = y_hat[lag:]
    m = min(n_plot, len(actual))

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
    plt.plot(np.arange(m), forecast[:m], 'x--', label='1-step Forecast')
    plt.title(f'One-Step Forecast vs Actual (first {m} points)')
    plt.xlabel('t')
    plt.legend()
    plt.grid(True)
    plt.show()

    res_var = np.var(actual[:m] - forecast[:m])
    print(f"1-step residual var       = {res_var:.4f}")

    return forecast

def h_step_forecast_arma_plot(y, e, phi, theta, h, y_actual=None):

    y = np.asarray(y)
    e = np.asarray(e)
    p, q = len(phi), len(theta)

    # prepare history
    y_hist = list(y)
    e_hist = list(e)
    forecasts = []

    for step in range(h):
        ar = sum(phi[i] * y_hist[-i-1] for i in range(p))
        ma = sum(theta[j] * e[-j-1] if step == 0 else 0 for j in range(q))
        y_next = ar + ma
        forecasts.append(y_next)
        y_hist.append(y_next)
        e_hist.append(0)

    # plot if true future provided
    if y_actual is not None:
        actual = np.asarray(y_actual)
        m = min(h, len(actual))
        plt.figure(figsize=(8, 3))
        plt.plot(np.arange(m), actual[:m],   'o-', label='Actual')
        plt.plot(np.arange(m), forecasts[:m], 'x--', label=f'{h}-step Forecast')
        plt.title(f'{h}-Step Forecast vs Actual')
        plt.xlabel('h')
        plt.legend()
        plt.grid(True)
        plt.show()

        fc_var = np.var(actual[:m] - forecasts[:m])
        print(f"{h}-step forecast var     = {fc_var:.4f}")

    return forecasts




# SARIMA

def forecast_sarima_one_step(data, t, phi, d, theta, seasonal_period, Phi=None, D=0, Theta=None):
    p, P = len(phi), len(Phi or [])
    y_hat = 0
    for i in range(1, p + 1):
        y_hat += phi[i - 1] * data[t - i]
    if Phi:
        for i in range(1, P + 1):
            y_hat += Phi[i - 1] * data[t - i * seasonal_period]
    return y_hat

def forecast_sarima_h_step(data, t0, hmax, phi, d, theta, seasonal_period, Phi=None, D=0, Theta=None):
    p, P = len(phi), len(Phi or [])
    y_pred = np.zeros(hmax)
    for h in range(1, hmax + 1):
        forecast = 0
        for i in range(1, p + 1):
            idx = t0 + h - i
            term = data[idx] if idx <= t0 else y_pred[idx - t0 - 1]
            forecast += phi[i - 1] * term
        if Phi:
            for i in range(1, P + 1):
                idx = t0 + h - i * seasonal_period
                term = data[idx] if idx <= t0 else y_pred[idx - t0 - 1]
                forecast += Phi[i - 1] * term
        y_pred[h - 1] = forecast
    return y_pred

def plot_sarima_forecasts(data, one_step_preds, t_start, h_preds, t0, hmax):
    # Plot one-step forecast
    actual_one_step = data[t_start + 1: t_start + 1 + len(one_step_preds)]
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(t_start + 1, t_start + 1 + len(actual_one_step)), actual_one_step, label="Actual", marker='o')
    plt.plot(np.arange(t_start + 1, t_start + 1 + len(one_step_preds)), one_step_preds, label="1-step Forecast", marker='x')
    plt.title("1-step Forecast vs Actual")
    plt.xlabel("Time Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot h-step forecast
    actual_h = data[t0 + 1: t0 + 1 + hmax]
    h_preds = h_preds[:len(actual_h)]
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(t0 + 1, t0 + 1 + len(actual_h)), actual_h, label="Actual", marker='o')
    plt.plot(np.arange(t0 + 1, t0 + 1 + len(h_preds)), h_preds, label=f"{hmax}-step Forecast", marker='x')
    plt.title(f"{hmax}-step Forecast from t = {t0}")
    plt.xlabel("Time Index")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print error variance
    residuals = actual_one_step - one_step_preds
    forecast_errors = actual_h - h_preds
    print(f"1-step Residual Variance: {np.var(residuals):.4f}")
    print(f"{hmax}-step Forecast Error Variance: {np.var(forecast_errors):.4f}")
    print(f"Variance Ratio (Test/Train): {np.var(forecast_errors) / np.var(residuals):.4f}")

#  BOX JENKINS

def estimate_impulse_response(u, y, K):
    N = len(u)
    Ru = np.array([np.sum(u[:N - tau] * u[tau:]) / (N - tau) for tau in range(K + 1)])
    Ruy = np.array([np.sum(u[:N - tau] * y[tau:]) / (N - tau) for tau in range(K + 1)])
    R_u_matrix = toeplitz(Ru[:K + 1])
    g_hat = np.linalg.solve(R_u_matrix, Ruy[:K + 1])
    return g_hat

# --- 2. GPAC Table Generator ---
def gpac_table_bj(acf_values, max_k=7, max_j=7):
    gpac = np.zeros((max_j, max_k))
    for j in range(max_j):
        for k in range(1, max_k + 1):
            try:
                D = np.array([[acf_values[abs(j + i - m)] for m in range(k)] for i in range(k)])
                N = D.copy()
                for i in range(k):
                    N[i, -1] = acf_values[j + i + 1]
                det_D = np.linalg.det(D)
                det_N = np.linalg.det(N)
                if np.isclose(det_D, 0):
                    gpac[j, k - 1] = np.nan if np.isclose(det_N, 0) else np.inf
                else:
                    gpac[j, k - 1] = det_N / det_D
            except Exception:
                gpac[j, k - 1] = np.nan
    return pd.DataFrame(gpac, index=[f"j={j}" for j in range(max_j)],
                        columns=[f"k={k + 1}" for k in range(max_k)])


def compute_residuals_bj(theta, y, u, nb, nf, nc, nd):
    N = len(y)
    y_g = np.zeros(N)
    e = np.zeros(N)

    b = np.r_[1, theta[:nb - 1]]
    f = np.r_[1, theta[nb - 1:nb - 1 + nf]]
    c = np.r_[1, theta[nb - 1 + nf:nb - 1 + nf + nc]]
    d = np.r_[1, -theta[nb - 1 + nf + nc:]]

    for t in range(max(len(b), len(f)), N):
        y_g[t] = sum(b[i] * u[t - i] for i in range(len(b))) - sum(f[j] * y_g[t - j] for j in range(1, len(f)))

    residual = y - y_g

    for t in range(max(len(c), len(d)), N):
        num = sum(d[j] * residual[t - j] for j in range(len(d)))
        den = sum(c[i] * e[t - i] for i in range(1, len(c)))
        e[t] = num - den

    return e

# --- 4. One-Step Forecast Function (Manual BJ) ---
def forecast_bj_1step(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    b1, f1, c1, d1 = theta
    N = len(y)
    yhat = []
    idx = []
    for t in range(start, start + steps):
        if t < 1 or t >= N - 1:
            continue
        val = b1 * u[t] - f1 * y[t] + c1 * e[t] - d1 * e[t - 1]
        yhat.append(val)
        idx.append(t)
    return np.array(yhat), idx

# --- 5. H-Step Forecast Function (recursive) ---
def forecast_bj_hstep(y, u, e, theta, nb, nf, nc, nd, steps=20, start=0):
    b1, f1, c1, d1 = theta
    yhat = []
    y_current = list(y[:start + 1])
    e_current = list(e[:start + 1])
    for h in range(steps):
        t = start + h
        val = b1 * u[t] - f1 * y_current[-1] + c1 * e_current[-1] - d1 * e_current[-2]
        yhat.append(val)
        y_current.append(val)
        e_current.append(0)
    return np.array(yhat)


