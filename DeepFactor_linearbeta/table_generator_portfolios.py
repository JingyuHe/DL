import pandas as pd
import numpy as np

##################################################
# Jianeng's original code to calculate R2 tables #
# Depreciated, using Yuanzhi's R version instead #
##################################################

# from DeepFunctions import *
from Tables import *

data_path = "~/DL_code/60chars"
result_path = "~/DL_code/results"

layer_list = [0, 1, 2, 3]
g_dim_list = [1, 5]
l1_v_list = [-3, -4, -5]
l2_v_list = [-3, -4, -5]
n_factor = 5

ff25 = pd.read_parquet(f"{data_path}/ff25.parquet")
ind49 = pd.read_parquet(f"{data_path}/ind49.parquet")
bisort = pd.read_parquet(f"{data_path}/bisort.parquet")
unisort = pd.read_parquet(f"{data_path}/unisort.parquet")

# Generate tables
stock_return = pd.read_parquet(f"{data_path}/winsorized_returns.parquet")[
    "excess_return"
].unstack(-1)
ff5 = pd.read_parquet(f"{data_path}/ff5.parquet")
capm = ff5[["Mkt-RF"]]
ipca = pd.read_parquet(f"{data_path}/ipca.parquet")
rppca = pd.read_parquet(f"{data_path}/rppca.parquet")
capm.index = np.arange(600)
ff5.index = np.arange(600)
beta_chars = pd.read_parquet(f"{data_path}/beta_chars.parquet")
beta_chars["intercept"] = 1
beta_chars = beta_chars[["intercept", "rank_bm", "rank_me", "rank_agr", "rank_op"]]

factors_dict = {"ff5": ff5, "ipca": ipca, "rppca": rppca}
portfolios_dict = {
    "ff25": ff25,
    "ind49": ind49,
    "bisort": bisort,
    "unisort": unisort,
    "stock": stock_return,
}


# =================================================================================================
print("Model Selection from CV ....... \n\n")
# =================================================================================================
metric_values = []
metric_name = []
factor_name = []
nf = []
nl = []
l1_list = []
l2_list = []
g_list = []
period = []
asset_list = []

for p_name, p_return in portfolios_dict.items():
    for f_name, f in factors_dict.items():
        print(f_name)
        if p_name == "stock":
            metric = stock_metrics(f, p_return, capm, cv_index=2, n_train=240)
        else:
            metric = portfolio_metrics(f, p_return, capm, cv_index=2, n_train=240)

        for m in ["total_r2", "predictive_r2", "sharpe"]:
            if (m == "sharpe") and (p_name != "stock"):
                continue
            for p in ["train", "test"]:
                factor_name.append(f_name)
                nf.append(np.nan)
                nl.append(np.nan)
                l1_list.append(np.nan)
                l2_list.append(np.nan)
                g_list.append(np.nan)
                metric_name.append(m)
                period.append(p)
                metric_values.append(metric[m][p])
                asset_list.append(p_name)

## Jingyu

cv_loss = []

for g_dim in g_dim_list:
    for n_layer in layer_list:
        for l1 in l1_v_list:
            for l2 in l2_v_list:
                cv0 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l2}_cv0.parquet")
                cv1 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l2}_cv1.parquet")
                cv = (cv0 + cv1) / 2.0
                cv = cv.iloc[99]
                cv["g_dim"] = int(g_dim)
                cv["n_layer"] = n_layer
                cv["n_factor"] = n_factor
                cv["l1"] = l1
                cv["l2"] = l2
                cv_loss.append(cv)

cv_loss = pd.DataFrame(cv_loss)

##



for p_name, p_return in portfolios_dict.items():
    for g_dim in g_dim_list:
        for n_layer in layer_list:
            print("g=", g_dim, "n_layer", n_layer)
            # select a best deep learning model for each layer
            tot_r2 = -1e10
            best_l1 = None
            best_l2 = None
            beta_f = None
            best_f_dim = None

            for l1 in l1_v_list:
                for l2 in l2_v_list:
                    deep_factors_0 = pd.read_parquet(
                        f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_{l2}_cv0.parquet"
                    )

                    deep_factors_1 = pd.read_parquet(
                        f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_{l2}_cv1.parquet"
                    )

                    for f in [i + 1 for i in range(n_factor)]:
                        if g_dim == 1:
                            g = capm.copy()
                            f_dim = 1 + f
                        elif g_dim == 5:
                            g = ff5.copy()
                            f_dim = 5 + f
                        factors_0 = pd.concat([deep_factors_0.iloc[:, :f], g], axis=1)
                        factors_1 = pd.concat([deep_factors_1.iloc[:, :f], g], axis=1)

                        if p_name == "stock":
                            metric0 = stock_metrics(
                                factors_0,
                                p_return,
                                capm,
                                cv_index=0,
                                raw_z=beta_chars,
                                n_train=240,
                            )
                            metric1 = stock_metrics(
                                factors_0,
                                p_return,
                                capm,
                                cv_index=1,
                                raw_z=beta_chars,
                                n_train=240,
                            )
                        else:
                            metric0 = portfolio_metrics(
                                factors_0, p_return, capm, cv_index=0, n_train=240,
                            )
                            metric1 = portfolio_metrics(
                                factors_0, p_return, capm, cv_index=1, n_train=240,
                            )

                        current = metric0["total_r2"]["test"] + metric1["total_r2"]["test"]
                        if current > tot_r2:
                            tot_r2 = current
                            best_l1 = l1
                            best_l2 = l2
                            best_f = f
                            best_f_dim = f_dim

            deep_factors = pd.read_parquet(
                f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{best_l1}_{best_l2}_cv2.parquet"
            )

            factors = pd.concat([deep_factors.iloc[:, :best_f], g], axis=1)
            if p_name == "stock":
                metric = stock_metrics(
                    factors,
                    stock_return,
                    capm,
                    cv_index=2,
                    raw_z=beta_chars,
                    n_train=240,
                )
            else:
                metric = portfolio_metrics(
                    factors, p_return, capm, cv_index=2, n_train=240
                )

            for m in ["total_r2", "predictive_r2", "sharpe"]:
                if (m == "sharpe") and (p_name != "stock"):
                    continue
                for p in ["train", "test"]:
                    factor_name.append("deep_factor")
                    nf.append(best_f)
                    nl.append(n_layer)
                    l1_list.append(best_l1)
                    l2_list.append(best_l2)
                    g_list.append(g_dim)
                    metric_name.append(m)
                    period.append(p)
                    metric_values.append(metric[m][p])
                    asset_list.append(p_name)


metrics_table = pd.DataFrame(
    {
        "factor": factor_name,
        "n_factor": nf,
        "n_layer": nl,
        "regularization": l1_list,
        "regularization_beta": l2_list,
        "g": g_list,
        "metric": metric_name,
        "period": period,
        "value": metric_values,
        "asset": asset_list,
    }
)

metrics_table.to_parquet(f"{result_path}/metrics.parquet")
print("Done!")

