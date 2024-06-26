import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Layer


class Norm(Layer):
    def __init__(self):
        super(Norm, self).__init__(name="Norm")

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=1, keepdims=True)
        return (inputs - mean) / (tf.math.sqrt(var) + 0.00001)


class Weight(Layer):
    def __init__(self, params=[50, 5]):
        super(Weight, self).__init__(name="Weight")
        self.param = params

    def call(self, inputs):
        x = -self.param[0] * tf.exp(-self.param[1] * inputs)
        y = -self.param[0] * tf.exp(self.param[1] * inputs)
        positive_weight = tf.keras.layers.Softmax(axis=[0, 2])(x)
        negative_weight = tf.keras.layers.Softmax(axis=[0, 2])(y)
        return positive_weight - negative_weight


class Factor(Layer):
    def __init__(self):
        super(Factor, self).__init__(name="DeepFactor")

    def call(self, weight, ret):
        w_transpose = tf.transpose(weight, perm=[0, 2, 1])
        return tf.squeeze(tf.matmul(w_transpose, ret))


class Beta_Factor(Layer):
    def __init__(self, n_beta, n_factor):
        super(Beta_Factor, self).__init__(name="Beta_Factor")
        self.b = self.add_weight(
            shape=((n_beta + 1), n_factor), initializer="random_normal", trainable=True
        )

    def call(self, beta_char, factor):
        char_intercept = tf.pad(beta_char, [[0, 0], [0, 0], [1, 0]], constant_values=1)
        new_char = char_intercept @ self.b
        new_factor = tf.expand_dims(factor, axis=-1)
        return tf.squeeze(new_char @ new_factor), new_char, new_factor


class DeepFactorModel(Model):
    def __init__(
        self,
        n_stock,
        n_portfolio,
        n_beta,
        p,
        g_dim,
        hidden_sizes,
        beta_hidden_sizes,
        l1_lam=0.1,
        l1_lam_beta=0.1,
        activation="tanh",
        dropout=0.5,
        weight_params=[50, 5],
    ):
        super(DeepFactorModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.n_portfolio = n_portfolio
        self.avg_loss = tf.keras.metrics.Mean("loss", dtype=tf.float32)
        self.mse_metric = tf.keras.metrics.MeanSquaredError("mse", dtype=tf.float32)

        def l1_reg(weight_matrix):
            tot_l1 = tf.math.reduce_sum(tf.math.abs(weight_matrix))
            diag_l1 = tf.math.reduce_sum(
                tf.math.abs(tf.linalg.diag_part(weight_matrix))
            )
            return l1_lam * (tot_l1 - diag_l1)

        def l1_beta_reg(weight_matrix):
            tot_l1 = tf.math.reduce_sum(tf.math.abs(weight_matrix))
            diag_l1 = tf.math.reduce_sum(
                tf.math.abs(tf.linalg.diag_part(weight_matrix))
            )
            return l1_lam_beta * (tot_l1 - diag_l1)

        # construct layers
        self.input_char = InputLayer((n_stock, p))
        self.input_r = InputLayer(n_stock)
        self.input_g = InputLayer(g_dim)
        self.input_beta_char = InputLayer(n_beta)

        self.characteristics_layers = []
        for layer_size in self.hidden_sizes[:-1]:
            self.characteristics_layers.append(
                Dense(layer_size, activation=activation, kernel_regularizer=l1_reg)
            )
            self.characteristics_layers.append(Dropout(dropout))
        self.characteristics_layers.append(
            Dense(self.hidden_sizes[-1], activation=activation)
        )

        self.norm_layer = Norm()
        self.weight_layer = Weight(weight_params)
        self.factor_layer = Factor()

        self.beta_layers = []

        # All nonlinear with 0.1 dropout after final layer
        for layer_size in beta_hidden_sizes:
            self.beta_layers.append(
                Dense(
                    layer_size,
                    activation=activation,
                    kernel_regularizer=l1_beta_reg,
                )
            )
        # self.beta_layers.append(Dropout(0.1))

        # # First 2 layers nonlinear and final linear layer, without dropout
        # beta_count = 0
        # for layer_size in beta_hidden_sizes:
        #     if beta_count in [2]:
        #         print(layer_size)
        #         self.beta_layers.append(
        #             Dense(
        #                 layer_size,
        #                 activation="linear",
        #                 kernel_regularizer=l1_beta_reg,
        #             )
        #         )
        #     else:
        #         self.beta_layers.append(
        #             Dense(
        #                 layer_size,
        #                 activation=activation,
        #                 kernel_regularizer=l1_beta_reg,
        #             )
        #         )
        #     beta_count += 1

        self.beta_f_interaction = Beta_Factor(
            n_beta=beta_hidden_sizes[-1], n_factor=hidden_sizes[-1] + g_dim
        )

    def call(self, inputs, training=True):
        Z, stock_ret, additional_factors, beta_char = inputs
        Z = self.input_char(Z)
        ret = self.input_r(stock_ret)
        g = self.input_g(additional_factors)
        beta = self.input_beta_char(beta_char)

        for layer in self.characteristics_layers:
            Z = layer(Z, training=training)
        normalized_Z = self.norm_layer(Z)

        for layer in self.beta_layers:
            beta = layer(beta) #, training=training

        weight = self.weight_layer(normalized_Z)
        deep_factor = tf.reshape(
            self.factor_layer(weight, ret), [-1, self.hidden_sizes[-1]]
        )
        f_g = tf.concat([deep_factor, g], axis=1)
        r, deep_beta, factor = self.beta_f_interaction(beta, f_g) #, training=training

        return [deep_factor, Z, r, deep_beta, factor]

    def train_step(self, data):
        x, y = data
        factor_x = x[0]
        beta_x = x[-1]
        with tf.GradientTape() as tape, tf.GradientTape() as factor_tape, tf.GradientTape() as beta_tape:
            factor_tape.watch(factor_x)
            beta_tape.watch(beta_x)
            _, _, beta_f, _, _ = self(x, training=True)
            loss = self.compiled_loss(y, beta_f, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.avg_loss.update_state(loss)
        self.mse_metric.update_state(y, beta_f)
        gradients_factor = factor_tape.gradient(loss, factor_x)
        gradients_beta = beta_tape.gradient(loss, beta_x)

        return {"loss": self.avg_loss.result(), 
                "mse": self.mse_metric.result(), 
                "gradients": gradients,
                "gradients_factor": gradients_factor,
                "gradients_beta": gradients_beta}

    @property
    def metrics(self):
        return [self.avg_loss, self.mse_metric]


def get_model_prediction(model, data, batch=60):
    # in case GPU memory full
    n = data[0].shape[0]
    splits = np.append(np.arange(start=0, stop=n, step=batch), n)

    in_factors = []
    in_chars = []
    in_r = []
    in_beta = []
    in_factor = []
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1]
        data_subset = [x[start:end] for x in data]
        factor_subset, char_subset, r, beta, factor = model(data_subset, training=False)
        in_factors.append(np.array(factor_subset).reshape(-1, 1))
        in_chars.append(np.array(char_subset))
        in_r.append(np.array(r))
        in_beta.append(np.array(beta))
        in_factor.append(np.array(factor))
    in_factors = np.vstack(in_factors)
    in_chars = np.vstack(in_chars)
    in_r = np.vstack(in_r)
    in_beta = np.vstack(in_beta)
    in_factor = np.vstack(in_factor)

    return in_factors, in_chars, in_r, in_beta, in_factor


def sequential_deep_factor(
    result_path,
    input_data,
    n_layer,
    n_factor,
    g_dim,
    benchmark,
    n_beta_char,
    beta_hidden_sizes,
    l1_lam,
    l1_lam_beta,
    l1_lam_log,
    l1_lam_beta_log,
    n_train=360,
    n_test=120,
    cv_index=0,
    epoch=100,
    batch_size=60,
    learning_rate=0.002,
    verbose=1,
    n_stock=3000,
):
    if cv_index == 0:
        train_idx = [i for i in range(n_train)]
        test_idx = [i + n_train for i in range(n_train)]
    elif cv_index == 1:
        test_idx = [i for i in range(n_train)]
        train_idx = [i + n_train for i in range(n_train)]
    else:
        train_idx = [i for i in range(n_train * 2)]
        test_idx = [(i + n_train * 2) for i in range(n_test)]

    # get data
    Tn, p = input_data["characteristics"].shape
    T = int(Tn / n_stock)
    if cv_index == 2:
        assert len(train_idx) + len(test_idx) == T
    _, p_beta = input_data["beta_chars"].shape

    r = input_data["stock_returns"].reshape((T, n_stock))[:, :, np.newaxis]
    Z = input_data["characteristics"].reshape(-1, n_stock, p)

    beta = input_data["beta_chars"].reshape(-1, n_stock, p_beta)

    train_y = np.squeeze(r[train_idx])
    n_portfolio = n_stock
    train_r, test_r = r[train_idx], r[test_idx]
    train_Z, test_Z = Z[train_idx], Z[test_idx]
    train_g, test_g = (
        input_data["ff_factors"][train_idx, :g_dim].reshape(-1, g_dim),
        input_data["ff_factors"][test_idx, :g_dim].reshape(-1, g_dim),
    )
    train_beta_char, test_beta_char = beta[train_idx], beta[test_idx]
    hidden_sizes = [p] * n_layer + [1]

    # HACK
    if cv_index < 2:
        T = 2 * n_train

    # train model
    train_factors = []
    test_factors = []
    train_characteristics = []
    test_characteristics = []
    loss_sequence = []
    mse_sequence = []
    gradient_factor_sequence = []
    gradient_beta_sequence = []

    for idx in range(n_factor):
        print("Factor Index:", idx)
        model = DeepFactorModel(
            n_stock,
            n_portfolio,
            n_beta_char,
            p,
            g_dim,
            hidden_sizes,
            beta_hidden_sizes,
            l1_lam,
            l1_lam_beta,
        )
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
        )

        train_data = (train_Z, train_r, train_g, train_beta_char)
        test_data = (test_Z, test_r, test_g, test_beta_char)
        history = model.fit(
            train_data, train_y, batch_size=batch_size, epochs=epoch, verbose=verbose
        )

        # in case GPU memory full
        in_factors, in_chars, in_r, in_beta, in_factor = get_model_prediction(
            model, train_data, batch=60
        )
        out_factors, out_chars, out_r, out_beta, out_factor = get_model_prediction(
            model, test_data, batch=60
        )

        loss = np.array(history.history["loss"])
        mse = np.array(history.history["mse"])
        gradient_factor = np.abs(history.history["gradients_factor"][-1]).sum(axis=(0,1))
        gradient_factor = gradient_factor/np.sum(gradient_factor)
        gradient_beta = np.abs(history.history["gradients_beta"][-1]).sum(axis=(0,1))
        gradient_beta = gradient_beta/np.sum(gradient_beta)

        train_factors.append(in_factors)
        test_factors.append(out_factors)
        train_characteristics.append(in_chars)
        test_characteristics.append(out_chars)
        loss_sequence.append(loss)
        mse_sequence.append(mse)
        gradient_factor_sequence.append(gradient_factor)
        gradient_beta_sequence.append(gradient_beta)

        g_dim += 1
        train_g = np.hstack([train_g, train_factors[-1]])
        test_g = np.hstack([test_g, test_factors[-1]])

        rhat = pd.DataFrame(
            np.vstack([in_r, out_r]), columns=[f"rank_{i}" for i in range(n_stock)]
        )

        rhat.to_parquet(
            f"{result_path}/rhat{benchmark}_{n_layer}_{n_factor}_{l1_lam_log}_{l1_lam_beta_log}_{idx}_cv{cv_index}.parquet"
        )

    train_factors = np.hstack(train_factors)
    test_factors = np.hstack(test_factors)
    loss_sequence = np.vstack(loss_sequence)
    mse_sequence = np.vstack(mse_sequence)
    gradient_factor_sequence = np.vstack(gradient_factor_sequence)
    gradient_beta_sequence = np.vstack(gradient_beta_sequence)
    train_characteristics = np.stack(train_characteristics, -1).reshape(
        -1, n_stock, n_factor
    )
    test_characteristics = np.stack(test_characteristics, -1).reshape(
        -1, n_stock, n_factor
    )

    deep_factors = pd.DataFrame(np.vstack([train_factors, test_factors]))
    deep_factors.columns = [f"DF_{i}" for i in range(n_factor)]
    loss_sequence = pd.DataFrame(loss_sequence.T)
    loss_sequence.columns = [f"Loss_{i}" for i in range(n_factor)]
    mse_sequence = pd.DataFrame(mse_sequence.T)
    mse_sequence.columns = [f"MSE_{i}" for i in range(n_factor)]
    gradient_factor_sequence = pd.DataFrame(gradient_factor_sequence.T)
    gradient_factor_sequence.columns = [f"DF_{i}" for i in range(n_factor)]
    gradient_beta_sequence = pd.DataFrame(gradient_beta_sequence.T)
    gradient_beta_sequence.columns = [f"DF_{i}" for i in range(n_factor)]
    deep_chars = np.vstack([train_characteristics, test_characteristics])
    deep_chars = pd.concat(
        {i: pd.DataFrame(deep_chars[i]) for i in range(T)}, names=["month"]
    )
    deep_chars.columns = [f"DC_{i}" for i in range(n_factor)]

    deep_beta = np.vstack([in_beta, out_beta])
    deep_beta = pd.concat(
        {i: pd.DataFrame(deep_beta[i]) for i in range(T)}, names=["month"]
    )
    deep_beta.columns = [f"DB_{i}" for i in range(g_dim)]  # beta for [g, f]

    rhat = pd.DataFrame(
        np.vstack([in_r, out_r]), columns=[f"rank_{i}" for i in range(n_stock)]
    )

    deep_f_g = np.vstack([in_factor, out_factor])
    deep_f_g = pd.concat(
        {i: pd.DataFrame(deep_f_g[i].T) for i in range(T)}, names=["month"]
    )

    return (
        deep_factors,
        deep_chars,
        deep_beta,
        rhat,
        loss_sequence,
        mse_sequence,
        deep_f_g,
        gradient_factor_sequence,
        gradient_beta_sequence,
    )

