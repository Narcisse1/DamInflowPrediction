import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers  # Explicitly use tensorflow.keras.layers
from tensorflow.keras import models  # Explicitly use tensorflow.keras.models
from tensorflow.keras import callbacks  # Explicitly use tensorflow.keras.callbacks
from tensorflow.keras import metrics  # Explicitly use tensorflow.keras.metrics
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import warnings

# Suppress Matplotlib warnings for PostScript transparency
warnings.filterwarnings("ignore", category=UserWarning, message="The PostScript backend does not support transparency")

# Suppress TensorFlow oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Set plotting and random seed settings
plt.style.use('fivethirtyeight')
plt.rcParams['figure.autolayout'] = True
sns.set(style="whitegrid", palette="muted", font_scale=1.5)
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
register_matplotlib_converters()

# Define hyperparameters
HORIZON = 1  # Predict next day
WINDOW_SIZE = 7  # Use previous week of data

# Load and preprocess data
df = pd.read_csv("lom_pangar_inflow.csv", engine="python", parse_dates=[0],
                 index_col="Date", date_format="%d-%m-%y")  # Use date_format to avoid warning
data = pd.DataFrame(df["inflow"]).rename(columns={"inflow": "inflow"})

# Plot initial data
data.plot(figsize=(30, 15))
plt.ylabel("lom_pangar_inflow")
plt.title("lom_pangar_inflow from 2015 to 2020", fontsize=16)
plt.legend(fontsize=14)
plt.show()

# Get timesteps and inflow
timesteps = np.array(data.index)
inflow = data["inflow"].to_numpy()

# Create train/test splits
split_size = int(len(data) * 0.8)  # 80% train, 20% test
x_train, y_train = timesteps[:split_size], data[:split_size]
x_test, y_test = timesteps[split_size:], data[split_size:]
print(len(x_train), len(x_test), len(y_train), len(y_test))

# Plot train/test splits
plt.figure(figsize=(30, 15))
plt.scatter(x_train, y_train, s=6, label="Train data")
plt.scatter(x_test, y_test, s=6, label="Test data")
plt.xlabel("Date")
plt.ylabel("inflow")
plt.legend(fontsize=14)
plt.show()

# Define plotting function
def plot_time_series(timesteps, values, format='eps', dpi=1200, start=0, end=None, label=None):
    """
    Plot timesteps against values.
    """
    plt.plot(timesteps[start:end], values[start:end], label=label)
    plt.xlabel("Date", fontsize=26)
    plt.ylabel("Daily inflow in m^3/s", fontsize=26)
    if label:
        plt.legend(fontsize=26)
    plt.grid(True)
    if format and dpi:
        plt.savefig(f"plot_{label}.{format}", format=format, dpi=dpi)

# Create naive forecast
naive_forecast = y_test[:-1]
print(naive_forecast[:10], naive_forecast[-10:])

# Plot naive forecast
plt.figure(figsize=(30, 15))
plot_time_series(timesteps=x_train, values=y_train, format='eps', dpi=1400, label="Train data")
plot_time_series(timesteps=x_test, values=y_test, format='eps', dpi=1400, label="Test data")
plot_time_series(timesteps=x_test[1:], values=naive_forecast, format='eps', dpi=1400, label="naive_forecast")
plt.show()

# Zoomed naive plot
plt.figure(figsize=(30, 15))
offset = 0
plot_time_series(timesteps=x_test, values=y_test, format='eps', dpi=1400, start=offset, label="Test data")
plot_time_series(timesteps=x_test[1:], values=naive_forecast, format='eps', dpi=1400, start=offset, label="naive_forecast")
plt.show()

# Define MASE metric
def mean_absolute_scaled_error(y_true, y_pred):
    """
    Implement MASE (assuming no seasonality of data).
    """
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1]))
    return mae / mae_naive_no_season

# Define evaluation function
def evaluate_preds(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    mae_metric = metrics.MeanAbsoluteError()
    mse_metric = metrics.MeanSquaredError()
    mape_metric = metrics.MeanAbsolutePercentageError()
    mae_metric.update_state(y_true, y_pred)
    mse_metric.update_state(y_true, y_pred)
    mape_metric.update_state(y_true, y_pred)
    mae = mae_metric.result()
    mse = mse_metric.result()
    rmse = tf.sqrt(mse)
    mape = mape_metric.result()
    mase = mean_absolute_scaled_error(y_true, y_pred)
    nse = 1 - (tf.reduce_sum((y_pred - y_true) ** 2) / tf.reduce_sum((y_true - tf.reduce_mean(y_true)) ** 2))
    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
        "mape": mape.numpy(),
        "mase": mase.numpy(),
        "nse": nse.numpy()
    }

# Evaluate naive forecast
naive_results = evaluate_preds(y_true=y_test[1:], y_pred=naive_forecast)
print("naive_results")
print(naive_results)

# Windowing functions
def get_labelled_windows(x, horizon=1):
    """
    Creates labels for windowed dataset.
    """
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    """
    Turns a 1D array into a 2D array of sequential windows of window_size.
    """
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    windowed_array = x[window_indexes]
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels

# Create windowed dataset
full_windows, full_labels = make_windows(inflow, window_size=WINDOW_SIZE, horizon=HORIZON)
print(len(full_windows), len(full_labels))

# View first and last 3 windows/labels
for i in range(3):
    print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")
for i in range(-3, 0):
    print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")

# Create train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1 - test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
print(len(train_windows), len(test_windows), len(train_labels), len(test_labels))
print(train_windows[:5], train_labels[:5])
print(np.array_equal(np.squeeze(train_labels[:-HORIZON-1]), y_train[WINDOW_SIZE:]))

# Create ModelCheckpoint callback
def create_model_checkpoint(model_name, save_path="model_experiments", save_weights_only=False):
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, f"{model_name}.keras") if not save_weights_only else os.path.join(save_path, f"{model_name}_weights.weights.h5")
    return callbacks.ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True, save_weights_only=save_weights_only)

# Function to make predictions
def make_preds(model, input_data):
    forecast = model.predict(input_data)
    return tf.squeeze(forecast)

# Model 1: Dense model
model_1 = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON, activation="linear")
], name="model_1_dense")

model_1.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_1.fit(x=train_windows, y=train_labels, epochs=200, verbose=1, batch_size=128,
            validation_data=(test_windows, test_labels), callbacks=[create_model_checkpoint(model_name=model_1.name)])

model_1.evaluate(test_windows, test_labels)
model_1 = models.load_model("model_experiments/model_1_dense.keras")
model_1.evaluate(test_windows, test_labels)

model_1_preds = make_preds(model_1, test_windows)
print(len(model_1_preds), model_1_preds[:10])
model_1_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_1_preds)
print("model_1_results")
print(model_1_results)

plt.figure(figsize=(30, 15))
plot_time_series(timesteps=x_test[-len(test_windows):], values=test_labels[:, 0], format='eps', dpi=1400, start=0, label="inflow")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_1_preds, format='eps', dpi=1400, start=0, label="Dense model")
plt.show()

# Model 4: Conv1D model
train_windows = tf.cast(train_windows, tf.float32)
test_windows = tf.cast(test_windows, tf.float32)
train_labels = tf.cast(train_labels, tf.float32)
test_labels = tf.cast(test_labels, tf.float32)

def build_model_4():
    model = models.Sequential([
        layers.Lambda(lambda x: tf.expand_dims(x, axis=1)),
        layers.Conv1D(filters=128, kernel_size=5, padding="causal", activation="relu"),
        layers.Dense(1)
    ], name="model_4_conv1D")
    return model

model_4 = build_model_4()
model_4.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_4.fit(train_windows, train_labels, batch_size=128, epochs=100, verbose=0,
            validation_data=(test_windows, test_labels), callbacks=[create_model_checkpoint(model_name=model_4.name, save_weights_only=True)])

model_4.summary()
model_4 = build_model_4()
model_4.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_4.build(input_shape=(None, WINDOW_SIZE))
model_4.load_weights("model_experiments/model_4_conv1D_weights.weights.h5")
model_4.evaluate(test_windows, test_labels)

model_4_preds = make_preds(model_4, test_windows)
print(model_4_preds[:10])
model_4_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_4_preds)
print(model_4_results)

plt.figure(figsize=(30, 15))
plot_time_series(timesteps=x_test[-len(test_windows):], values=test_labels[:, 0], format='eps', dpi=1400, start=0, label="inflow")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_4_preds, format='eps', dpi=1400, start=0, label="Conv1D Model")
plt.show()

# Model 5: RNN (LSTM)
def build_model_5():
    inputs = layers.Input(shape=(WINDOW_SIZE,))  # Corrected shape
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    x = layers.LSTM(1000, activation="relu")(x)
    x = layers.Dense(100, activation="relu")(x)
    output = layers.Dense(1)(x)
    return models.Model(inputs=inputs, outputs=output, name="model_5_lstm")

model_5 = build_model_5()
model_5.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_5.fit(train_windows, train_labels, epochs=100, verbose=0, batch_size=128,
            validation_data=(test_windows, test_labels), callbacks=[create_model_checkpoint(model_name=model_5.name, save_weights_only=True)])

model_5 = build_model_5()
model_5.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_5.build(input_shape=(None, WINDOW_SIZE))
model_5.load_weights("model_experiments/model_5_lstm_weights.weights.h5")
model_5.evaluate(test_windows, test_labels)

model_5_preds = make_preds(model_5, test_windows)
print(model_5_preds[:10])
model_5_results = evaluate_preds(y_true=tf.squeeze(test_labels), y_pred=model_5_preds)
print("model_5_results")
print(model_5_results)

plt.figure(figsize=(30, 15))
plot_time_series(timesteps=x_test[-len(test_windows):], values=test_labels[:, 0], format='eps', dpi=1400, start=0, label="inflow")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_5_preds, format='eps', dpi=1400, start=0, label="LSTM model")
plt.show()

# Combined plot for Models 1, 4, 5
offset = 150
plt.figure(figsize=(40, 30))
plot_time_series(timesteps=x_test[-len(test_windows):], values=test_labels[:, 0], format='eps', dpi=1400, start=offset, label="Lom pangar inflow")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_1_preds, format='eps', dpi=1400, start=offset, label="Dense model")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_4_preds, format='eps', dpi=1400, start=offset, label="Conv1D Model")
plot_time_series(timesteps=x_test[-len(test_windows):], values=model_5_preds, format='eps', dpi=1400, start=offset, label="LSTM model")
plt.show()

# Model 6: Multivariate Dense
data_block = df[["inflow", "precipitation"]]
scaled_inflow_precipitation_df = pd.DataFrame(minmax_scale(data_block[["inflow", "precipitation"]]),
                                             columns=data_block.columns, index=data_block.index)
scaled_inflow_precipitation_df.plot(figsize=(10, 7))
plt.show()

data_windowed = data_block.copy()
for i in range(WINDOW_SIZE):
    data_windowed[f"inflow+{i+1}"] = data_windowed["inflow"].shift(periods=i+1)

x = data_windowed.dropna().drop("inflow", axis=1).astype(np.float32)
y = data_windowed.dropna()["inflow"].astype(np.float32)
print(x.head())
print(y.head())

split_size = int(len(x) * 0.8)
x_train, y_train = x[:split_size], y[:split_size]
x_test, y_test = x[split_size:], y[split_size:]
print(len(x_train), len(y_train), len(x_test), len(y_test))

model_6 = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(HORIZON)
], name="model_6_dense_multivariate")

model_6.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae"])
model_6.fit(x_train, y_train, epochs=100, batch_size=128, verbose=0,
            validation_data=(x_test, y_test), callbacks=[create_model_checkpoint(model_name=model_6.name)])

model_6 = models.load_model("model_experiments/model_6_dense_multivariate.keras")
model_6.evaluate(x_test, y_test)

model_6_preds = tf.squeeze(model_6.predict(x_test))
print(model_6_preds[:10])
model_6_results = evaluate_preds(y_true=y_test, y_pred=model_6_preds)
print("model_6_results")
print(model_6_results)

data_block_final = data_block[-model_6_preds.shape[0]:].sort_index(ascending=True)
data_block_final["model_6_preds"] = model_6_preds
data_block_final[["inflow", "model_6_preds"]].plot()
plt.xlabel("Date", fontsize=26)
plt.ylabel("Daily inflow in m^3/s", fontsize=26)
plt.legend(fontsize=26)
plt.show()

# Ensemble Model
data_nbeats = data_block.copy()
for i in range(WINDOW_SIZE):
    data_nbeats[f"inflow+{i+1}"] = data_nbeats["inflow"].shift(periods=i+1)

X = data_nbeats.dropna().drop("inflow", axis=1)
y = data_nbeats.dropna()["inflow"]

split_size = int(len(X) * 0.8)
x_train, y_train = X[:split_size], y[:split_size]
x_test, y_test = X[split_size:], y[split_size:]
print(len(x_train), len(y_train), len(x_test), len(y_test))

train_features_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)
test_features_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

BATCH_SIZE = 1024
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def get_ensemble_models(horizon=HORIZON, train_data=train_dataset, test_data=test_dataset,
                        num_iter=5, num_epochs=1000, loss_fns=["mae", "mse", "mape"]):
    ensemble_models = []
    for i in range(num_iter):
        for loss_function in loss_fns:
            print(f"Optimizing model by reducing: {loss_function} for {num_epochs} epochs, model number: {i}")
            model = models.Sequential([
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(128, kernel_initializer="he_normal", activation="relu"),
                layers.Dense(horizon)
            ], name=f"ensemble_model_{i}_{loss_function}")
            model.compile(loss=loss_function, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["mae", "mse"])
            model.fit(train_data, epochs=num_epochs, verbose=0, validation_data=test_data,
                      callbacks=[
                          callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                          callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1),
                          create_model_checkpoint(model_name=f"ensemble_model_{i}_{loss_function}")
                      ])
            ensemble_models.append(model)
    return ensemble_models

ensemble_models = get_ensemble_models(num_iter=5, num_epochs=1000)

def make_ensemble_preds(ensemble_models, data):
    ensemble_preds = []
    for model in ensemble_models:
        preds = model.predict(data)
        ensemble_preds.append(preds)
    return tf.constant(tf.squeeze(ensemble_preds))

ensemble_preds = make_ensemble_preds(ensemble_models=ensemble_models, data=test_dataset)
print(ensemble_preds)

ensemble_results = evaluate_preds(y_true=y_test, y_pred=np.median(ensemble_preds, axis=0))
print("ensemble_results")
print(ensemble_results)

def get_upper_lower(preds):
    std = tf.math.reduce_std(preds, axis=0)
    interval = 1.96 * std
    preds_mean = tf.reduce_mean(preds, axis=0)
    lower, upper = preds_mean - interval, preds_mean + interval
    return lower, upper

lower, upper = get_upper_lower(preds=ensemble_preds)
ensemble_median = np.median(ensemble_preds, axis=0)

plt.figure(figsize=(30, 20))
plt.plot(x_test.index[offset:], y_test[offset:], "g", label="Test Data")
plt.plot(x_test.index[offset:], ensemble_median[offset:], "k-", label="Ensemble Median")
plt.xlabel("Date")
plt.ylabel("Lom_pangar_inflow")
plt.fill_between(x_test.index[offset:], lower[offset:], upper[offset:], label="Prediction Intervals")
plt.legend(loc="upper right", fontsize=16)
plt.show()