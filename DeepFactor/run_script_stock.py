import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import time
import os
from DeepFunctions_v2 import *

print("TensorFlow version:", tf.__version__)


physical_devices = tf.config.list_physical_devices("GPU")
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

data_path = "~/DL_code/60chars"
result_path = "~/DL_code/results"

characteristics = pd.read_parquet(f"{data_path}/60chars.parquet").values.astype(
    np.float32
)
stock_return = pd.read_parquet(f"{data_path}/winsorized_returns.parquet")[
    "excess_return"
].values.astype(np.float32)
ff_factors = pd.read_parquet(f"{data_path}/ff5.parquet").values.astype(np.float32)
beta_chars = pd.read_parquet(f"{data_path}/beta_chars.parquet").values.astype(
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
l1_v_list = [-5]
l1_beta_list = [-5]
n_factor = 5
n_beta = beta_chars.shape[-1]
beta_hidden_sizes = [20]

total = len(layer_list) * len(g_dim_list) * len(l1_v_list)
count = 0

start_time = time.time()


for n_layer, g_dim, l1, l1_beta in itertools.product(
    layer_list, g_dim_list, l1_v_list, l1_beta_list
):

    count += 1
    print(
        f"Task {count}/{total}: layer={n_layer}, g={g_dim}, l1={l1}, l1_beta={l1_beta}"
    )

    for j in [2, 0, 1]:
        deep_factors, deep_chars, deep_beta, rhat, loss = sequential_deep_factor(
            result_path,
            input_data,
            n_layer=n_layer,
            n_factor=n_factor,
            g_dim=g_dim,
            benchmark=g_dim,
            n_beta_char=n_beta,
            beta_hidden_sizes=beta_hidden_sizes,
            l1_lam=np.exp(l1),
            l1_lam_beta=np.exp(l1_beta),
            l1_lam_log = l1,
            l1_lam_beta_log = l1_beta,
            n_train=240,
            cv_index=j,
            epoch=EPOCH,
            batch_size=BATCH,
            verbose=verbose,
        )
        deep_factors.to_parquet(
            f"{result_path}/factors{g_dim}_{n_layer}_{n_factor}_{l1}_{l1_beta}_cv{j}.parquet"
        )
        deep_chars.to_parquet(
            f"{result_path}/chars{g_dim}_{n_layer}_{n_factor}_{l1}_{l1_beta}_cv{j}.parquet"
        )

        deep_beta.to_parquet(
            f"{result_path}/beta{g_dim}_{n_layer}_{n_factor}_{l1}_{l1_beta}_cv{j}.parquet"
        )

        # rhat.to_parquet(
        #     f"{result_path}/rhat{g_dim}_{n_layer}_{n_factor}_{l1}_{l1_beta}_cv{j}.parquet"
        # )

        loss.to_parquet(
            f"{result_path}/loss{g_dim}_{n_layer}_{n_factor}_{l1}_{l1_beta}_cv{j}.parquet"
        )

    print(
        f"-------------- {int((time.time() - start_time) / 60)} minutes -------------- \n\n"
    )
    start_time = time.time()
