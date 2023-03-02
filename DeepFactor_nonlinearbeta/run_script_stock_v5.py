import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import time
import os
from DeepFunctions_v5 import *
import random

### Set random seed
random_seed = 1234

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = "1"
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    os.environ['TF_CUDNN_DETERMINISTIC'] = "1"

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(random_seed)

print("TensorFlow version:", tf.__version__)


physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

data_path = "../60chars"
result_path = "results"

if result_path not in os.listdir():
    os.mkdir(result_path)

characteristics = pd.read_parquet(f"{data_path}/60chars.parquet").values.astype(
    np.float32
)
stock_return = pd.read_parquet(f"{data_path}/winsorized_returns.parquet")[
    "excess_return"
].values.astype(np.float32)
ff_factors = pd.read_parquet(f"{data_path}/ff5.parquet").values.astype(np.float32)
beta_chars = pd.read_parquet(f"{data_path}/60chars.parquet").values.astype(
    np.float32
)

input_data = {
    "characteristics": characteristics,
    "portfolios": stock_return,
    "stock_returns": stock_return,
    "ff_factors": ff_factors,
    "beta_chars": beta_chars,
}

EPOCH = 100
BATCH = 120
verbose = 0

layer_list = [0, 1, 2, 3]
g_dim_list = [1, 5]
l1_v_list = [-4, -5, -6]
l1_beta = -6
n_factor = 5
n_beta = beta_chars.shape[-1]
beta_hidden_sizes = [60, 16, 4]

total = len(layer_list) * len(g_dim_list) * len(l1_v_list)
count = 0

start_time = time.time()


for n_layer, g_dim, l1 in itertools.product(
    layer_list, g_dim_list, l1_v_list
):

    count += 1
    print(
        f"Task {count}/{total}: layer={n_layer}, g={g_dim}, l1={l1}, l1_beta={l1+l1_beta}"
    )

    for j in [2, 0, 1]:
        deep_factors, deep_chars, deep_beta, rhat, loss, mse, f_g, gradient_factor, gradient_beta = sequential_deep_factor(
            result_path,
            input_data,
            n_layer=n_layer,
            n_factor=n_factor,
            g_dim=g_dim,
            benchmark=g_dim,
            n_beta_char=n_beta,
            beta_hidden_sizes=beta_hidden_sizes,
            l1_lam=np.exp(l1),
            l1_lam_beta=np.exp(l1+l1_beta),
            l1_lam_log = l1,
            l1_lam_beta_log = l1+l1_beta,
            n_train=240,
            cv_index=j,
            epoch=EPOCH,
            batch_size=BATCH,
            verbose=verbose,
        )
        # deep_factors.to_parquet(
        #     f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_{l1}_cv{j}.parquet"
        # )
        deep_chars.to_parquet(
            f"{result_path}/chars{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.parquet"
        )
        # deep_beta.to_parquet(
        #     f"{result_path}/beta{g_dim}_{n_layer}_{n_factor}_{l1}_{l1}_cv{j}.parquet"
        # )
        loss.to_parquet(
            f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.parquet"
        )
        mse.to_parquet(
            f"{result_path}/mse{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.parquet"
        )

        # Double save csv format files
        deep_factors.to_csv(
            f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        deep_beta.to_csv(
            f"{result_path}/beta{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        loss.to_csv(
            f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        mse.to_csv(
            f"{result_path}/mse{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        f_g.to_csv(
            f"{result_path}/f_g{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        gradient_factor.to_csv(
            f"{result_path}/gradient_factor{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )
        gradient_beta.to_csv(
            f"{result_path}/gradient_beta{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv{j}.csv", index=False, encoding='utf_8_sig'
        )

    print(
        f"-------------- {int((time.time() - start_time) / 60)} minutes -------------- \n\n"
    )
    start_time = time.time()

##################################################
# find cv results, load .parquet directly
# if parquet files are saved, you may skip the above fitting procedures

cv_loss = []
cv_mse = []
for g_dim in g_dim_list:
    for n_layer in layer_list:
        for l1 in l1_v_list:
            cv0 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv0.parquet")
            cv1 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv1.parquet")
            mse0 = pd.read_parquet(f"{result_path}/mse{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv0.parquet")
            mse1 = pd.read_parquet(f"{result_path}/mse{g_dim}_{n_layer}_{n_factor}_{l1}_{l1+l1_beta}_cv1.parquet")
            cv = (cv0 + cv1) / 2.0
            cv = cv.iloc[99]
            mse_total = (mse0 + mse1) / 2.0
            mse_total = mse_total.iloc[99]
            cv["g_dim"] = int(g_dim)
            cv["n_layer"] = n_layer
            cv["n_factor"] = n_factor
            cv["l1"] = l1
            cv["l2"] = l1+l1_beta
            cv_loss.append(cv)
            mse_total["g_dim"] = int(g_dim)
            mse_total["n_layer"] = n_layer
            mse_total["n_factor"] = n_factor
            mse_total["l1"] = l1
            mse_total["l2"] = l1+l1_beta
            cv_mse.append(mse_total)

cv_loss = pd.DataFrame(cv_loss)
cv_mse = pd.DataFrame(cv_mse)

# save to csv file

cv_loss.to_csv(f"{result_path}/cv_results_nonlinearbeta.csv", index=False, encoding='utf_8_sig')
cv_mse.to_csv(f"{result_path}/cv_results_mse.csv", index=False, encoding='utf_8_sig')

frame_choose = []
factor_num = []
for g in g_dim_list:
    for layer in layer_list:
        temp_loss = cv_loss[(cv_loss['g_dim'] == g) & (cv_loss['n_layer'] == layer)].reset_index(drop=True)
        temp_loss = temp_loss.dropna(subset=['Loss_4'], axis=0)
        temp_loss['temp_col'] = 999999
        loss_mat = np.array(temp_loss[['Loss_4', 'temp_col']])
        x, y = np.where(loss_mat == np.min(loss_mat))
        frame_choose.append(temp_loss.loc[x,])
        
        # if y == 0:
        #     factor_n = 3
        # elif y == 1:
        #     factor_n = 4
        # elif y == 2:
        factor_n = 5
        factor_num.append(factor_n)
cv_loss = pd.concat(frame_choose).reset_index(drop=True)
cv_loss['min_n_factor'] = factor_num
# cv_loss['model_list'] = cv_loss.apply(lambda col: 'factors'+str(int(col['g_dim']))+'_'+str(int(col['n_layer']))+'_5_'+str(int(col['l1']))+'_'+str(int(col['l2']))+'_cv2_'+str(int(col['min_n_factor'])), axis=1)

cv_loss['model_list'] = cv_loss.apply(lambda col: 'factors'+str(int(col['g_dim']))+'_'+str(int(col['n_layer']))+'_5_'+str(int(col['l1']))+'_'+str(int(col['l2']))+'_cv2_5', axis=1)
cv_loss.to_csv(f"{result_path}/best_loss.csv", index=False, encoding='utf_8_sig')


frame_choose = []
factor_num = []
for g in g_dim_list:
    for layer in layer_list:
        temp_mse = cv_mse[(cv_mse['g_dim'] == g) & (cv_mse['n_layer'] == layer)].reset_index(drop=True)
        temp_mse = temp_mse.dropna(subset=['MSE_4'], axis=0)
        temp_mse['temp_col'] = 999999
        mse_mat = np.array(temp_mse[['MSE_4', 'temp_col']])
        x, y = np.where(mse_mat == np.min(mse_mat))
        frame_choose.append(temp_mse.loc[x,])
        
        # if y == 0:
        #     factor_n = 3
        # elif y == 1:
        #     factor_n = 4
        # elif y == 2:
        factor_n = 5
        factor_num.append(factor_n)
cv_mse = pd.concat(frame_choose).reset_index(drop=True)
cv_mse['min_n_factor'] = factor_num

cv_mse['model_list'] = cv_mse.apply(lambda col: 'factors'+str(int(col['g_dim']))+'_'+str(int(col['n_layer']))+'_5_'+str(int(col['l1']))+'_'+str(int(col['l2']))+'_cv2_5', axis=1)
cv_mse.to_csv(f"{result_path}/best_MSE.csv", index=False, encoding='utf_8_sig')
