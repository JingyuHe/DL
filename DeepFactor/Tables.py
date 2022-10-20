from concurrent.futures.thread import BrokenThreadPool
import numpy as np
import pandas as pd


def regression(y, x):
    return np.linalg.solve(x.T @ x, x.T @ y)


def regression_resid(y, x):
    beta = np.linalg.solve(x.T @ x, x.T @ y)[1:]
    return y - x[:, 1:] @ beta, beta


def factor_sharpe(f_in, f_out):
    lam = np.linalg.solve(f_in.cov(), f_in.mean())
    f_in = f_in.values @ lam
    f_out = f_out.values @ lam
    sharpe_in = np.mean(f_in) / np.std(f_in) * np.sqrt(12)
    sharpe_out = np.mean(f_out) / np.std(f_out) * np.sqrt(12)
    return sharpe_in, sharpe_out


def col_mse(x):
    return np.mean(np.nanmean(x ** 2, axis=0))


def total_R2(resid_1, resid_2):
    return 1 - np.nanmean(resid_1 ** 2) / np.nanmean(resid_2 ** 2)


def cs_R2(resid_1, resid_2):
    num = np.mean(np.nanmean(resid_1, axis=0) ** 2)
    den = np.mean(np.nanmean(resid_2, axis=0) ** 2)
    return 1 - num / den


def add_interaction(factors, macros):
    df = factors.copy()
    for x in macros:
        for y in factors:
            df[f"{x}_{y}"] = macros[x] * factors[y]
    return df


def rolling_regression_resid(y, x, start, window=60):
    resid = []
    beta = []
    for i in range(start, len(y)):
        beta.append(
            regression(
                y[max(i - 1 - window, 0) : (i - 1)], x[max(i - 1 - window, 0) : (i - 1)]
            )[1:]
        )
        resid.append(y[i] - x[i][1:] @ beta[-1])

    return np.array(resid), np.array(beta)


def stock_metrics(
    factors,
    stocks,
    market,
    cv_index,
    f_beta=None,
    n_train=240,
    n_test=120,
    n_stock=3000,
    raw_z=None,
):

    if cv_index == 0:
        train_idx = [i for i in range(n_train)]
        test_idx = [i + n_train for i in range(n_train)]
    elif cv_index == 1:
        train_idx = [i + n_train for i in range(n_train)]
        test_idx = [i for i in range(n_train)]
    else:
        train_idx = [i for i in range(n_train * 2)]
        test_idx = [(i + n_train * 2) for i in range(n_test)]

    x = factors.values
    y = stocks.values

    mkt = market.values
    if f_beta is not None:
        T = int(len(f_beta) / n_stock)
        beta = f_beta.values.reshape(T, n_stock, -1)
        x = x[..., np.newaxis]
    elif raw_z is not None:
        T = int(len(raw_z) / n_stock)
        beta_char = raw_z.values.reshape(T, n_stock, -1)
        rawf = np.tile(x[:, np.newaxis, :], [1, n_stock, 1])

        k = beta_char.shape[-1] * rawf.shape[-1]
        rx = np.zeros((T, n_stock, k))
        idx = 0
        for i in range(beta_char.shape[-1]):
            for j in range(rawf.shape[-1]):
                rx[:, :, idx] = beta_char[:, :, i] * rawf[:, :, j]
                idx += 1
        rx = rx[train_idx].reshape(-1, k)
        b = np.linalg.solve(rx.T @ rx, rx.T @ y[train_idx].reshape(-1, 1))
        beta = beta_char @ b.reshape(beta_char.shape[-1], rawf.shape[-1])
        x = x[..., np.newaxis]
    else:
        factors_with_intercept = factors.copy()
        cols = factors_with_intercept.columns
        factors_with_intercept["intercept"] = 1
        factors_with_intercept = factors_with_intercept[["intercept"] + list(cols)]

        _, beta = regression_resid(
            y[train_idx], factors_with_intercept.values[train_idx]
        )

    xa = np.mean(x[train_idx], axis=0)
    mkta = np.mean(mkt[train_idx])

    if (f_beta is not None) or (raw_z is not None):
        xa = np.tile(xa, [len(train_idx) + len(test_idx), 1, 1])
    else:
        xa = np.tile(xa, [len(train_idx) + len(test_idx), 1])
    metrics_dict = {}

    # Total R2 and Predictive R2
    if (f_beta is None) and (raw_z is None):
        resid_out = y[test_idx] - x[test_idx] @ beta
        resid_pred_out = y[test_idx] - xa[test_idx] @ beta
    else:
        resid_out = y[test_idx] - np.squeeze(beta[test_idx] @ x[test_idx])
        resid_pred_out = y[test_idx] - np.squeeze(beta[test_idx] @ xa[test_idx])

    bench_resid_out = y[test_idx] - mkt[test_idx]
    bench_resid_pred_out = y[test_idx] - mkta

    metrics_dict["total_r2"] = {
        "test": total_R2(resid_out, bench_resid_out),
    }
    metrics_dict["predictive_r2"] = {
        "test": total_R2(resid_pred_out, bench_resid_pred_out)
    }

    if cv_index == 2:
        if (f_beta is None) and (raw_z is None):
            resid_in = y[train_idx] - x[train_idx] @ beta
            resid_pred_in = y[train_idx] - xa[train_idx] @ beta
        else:
            resid_in = y[train_idx] - np.squeeze(beta[train_idx] @ x[train_idx])
            resid_pred_in = y[train_idx] - np.squeeze(beta[train_idx] @ xa[train_idx])

        bench_resid_in = y[train_idx] - mkt[train_idx]
        bench_resid_pred_in = y[train_idx] - mkta
        metrics_dict["total_r2"]["train"] = total_R2(resid_in, bench_resid_in)
        metrics_dict["predictive_r2"]["train"] = total_R2(
            resid_pred_in, bench_resid_pred_in
        )

    # Sharpe
    if cv_index == 2:
        sharpe_in, sharpe_out = factor_sharpe(
            factors.iloc[train_idx], factors.iloc[test_idx]
        )
        metrics_dict["sharpe"] = {
            "train": sharpe_in,
            "test": sharpe_out,
        }

    return metrics_dict


def portfolio_metrics(
    factors, portfolios, market, cv_index, n_train=240, n_test=120, n_stock=3000,
):

    if cv_index == 0:
        train_idx = [i for i in range(n_train)]
        test_idx = [i + n_train for i in range(n_train)]
    elif cv_index == 1:
        train_idx = [i + n_train for i in range(n_train)]
        test_idx = [i for i in range(n_train)]
    else:
        train_idx = [i for i in range(n_train * 2)]
        test_idx = [(i + n_train * 2) for i in range(n_test)]

    x = factors.values
    y = portfolios.values

    mkt = market.values

    factors_with_intercept = factors.copy()
    cols = factors_with_intercept.columns
    factors_with_intercept["intercept"] = 1
    factors_with_intercept = factors_with_intercept[["intercept"] + list(cols)]

    mkt_with_intercept = market.copy()
    cols = mkt_with_intercept.columns
    mkt_with_intercept["intercept"] = 1
    mkt_with_intercept = mkt_with_intercept[["intercept"] + list(cols)]

    _, beta = regression_resid(y[train_idx], factors_with_intercept.values[train_idx])

    _, bench_beta = regression_resid(y[train_idx], mkt_with_intercept.values[train_idx])

    xa = np.mean(x[train_idx], axis=0)
    mkta = np.mean(mkt[train_idx])
    xa = np.tile(xa, [len(train_idx) + len(test_idx), 1])
    metrics_dict = {}

    # Total R2, Cross-sectional R2 and Predictive R2
    resid_out = y[test_idx] - x[test_idx] @ beta
    resid_pred_out = y[test_idx] - xa[test_idx] @ beta
    bench_resid_out = y[test_idx] - mkt[test_idx] * bench_beta
    bench_resid_pred_out = y[test_idx] - mkta * bench_beta
    metrics_dict["total_r2"] = {
        "test": total_R2(resid_out, bench_resid_out),
    }
    metrics_dict["cs_r2"] = {"test": cs_R2(resid_out, bench_resid_out)}
    metrics_dict["predictive_r2"] = {
        "test": total_R2(resid_pred_out, bench_resid_pred_out)
    }

    if cv_index == 2:
        resid_in = y[train_idx] - x[train_idx] @ beta
        resid_pred_in = y[train_idx] - xa[train_idx] @ beta

        bench_resid_in = y[train_idx] - mkt[train_idx]
        bench_resid_pred_in = y[train_idx] - mkta * bench_beta
        metrics_dict["total_r2"]["train"] = total_R2(resid_in, bench_resid_in)
        metrics_dict["cs_r2"]["train"] = cs_R2(resid_in, bench_resid_in)
        metrics_dict["predictive_r2"]["train"] = total_R2(
            resid_pred_in, bench_resid_pred_in
        )

    return metrics_dict

