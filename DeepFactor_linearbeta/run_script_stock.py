import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import time
import os
from DeepFunctions_v2 import *

##################################################
# works well for tensorflow 2.10 version

print("TensorFlow version:", tf.__version__)

##################################################
# using GPU, set it as CPU if there is no GPU
physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# do not print warnings 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# set directory and load data
data_path = "~/DL_code/60chars"
result_path = "~/DL_code/results"
characteristics = pd.read_parquet(f"{data_path}/60chars.parquet").values.astype(
    np.float32
)
stock_return = pd.read_parquet(f"{data_path}/winsorized_returns.parquet")[
    "excess_return"
].values.astype(np.float32)
ff_factors = pd.read_parquet(f"{data_path}/ff5.parquet").values.astype(np.float32)

# variables for beta
beta_chars = pd.read_parquet(f"{data_path}/beta_chars.parquet").values.astype(
    np.float32
)

# summarize input data
input_data = {
    "characteristics": characteristics,
    "portfolios": stock_return,
    "stock_returns": stock_return,
    "ff_factors": ff_factors,
    "beta_chars": beta_chars,
}

##################################################
# training parameters
EPOCH = 100
BATCH = 120

# do not print
verbose = 0

# number of layers for the deep factor network
layer_list = [0, 1, 2, 3]
# number of augmented g factors
g_dim_list = [1, 5]
# candidates for L1 regularization on deep factor network
l1_v_list = [-3, -4, -5]
# number of factors
n_factor = 5
# number of x variables for beta, should match beta_chars
n_beta = 4

# total number of parameters
total = len(layer_list) * len(g_dim_list) * len(l1_v_list)
count = 0

##################################################
# run the main algorithm fitting neural networks, save results to .parquet files

start_time = time.time()

for n_layer, g_dim, l1 in itertools.product(layer_list, g_dim_list, l1_v_list,):

    count += 1
    print(f"Task {count}/{total}: layer={n_layer}, g={g_dim}, l1={l1}")

    for j in [2, 0, 1]:
        deep_factors, deep_chars, b, loss = sequential_deep_factor(
            input_data,
            n_layer=n_layer,
            n_factor=n_factor,
            g_dim=g_dim,
            n_beta_char=n_beta,
            l1_lam=np.exp(l1),
            n_train=240,
            cv_index=j,
            epoch=EPOCH,
            batch_size=BATCH,
            verbose=verbose,
        )
        deep_factors.to_parquet(
            f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_cv{j}.parquet"
        )
        deep_chars.to_parquet(
            f"{result_path}/chars{g_dim}_{n_layer}_{n_factor}_{l1}_cv{j}.parquet"
        )

        if not b.empty:
            b.to_parquet(
                f"{result_path}/b{g_dim}_{n_layer}_{n_factor}_{l1}_cv{j}.parquet"
            )

        loss.to_parquet(
            f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_cv{j}.parquet"
        )

    print(
        f"-------------- {int((time.time() - start_time) / 60)} minutes -------------- \n\n"
    )
    start_time = time.time()


##################################################
# find cv results, load .parquet directly
# if parquet files are saved, you may skip the above fitting procedures

cv_loss = []
for g_dim in g_dim_list:
    for n_layer in layer_list:
        for l1 in l1_v_list:
            cv0 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_cv0.parquet")
            cv1 = pd.read_parquet(f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_cv1.parquet")
            cv = (cv0 + cv1) / 2.0
            cv = cv.iloc[99]
            cv["g_dim"] = int(g_dim)
            cv["n_layer"] = n_layer
            cv["n_factor"] = n_factor
            cv["l1"] = l1
            cv_loss.append(cv)

cv_loss = pd.DataFrame(cv_loss)

# save to csv file

cv_loss.to_csv("cv_results.csv")

