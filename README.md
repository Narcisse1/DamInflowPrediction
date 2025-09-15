# Dam Inflow Prediction

This project focuses on predicting dam inflow using time series data and deep learning techniques. It leverages a Sequential Keras model to forecast future inflow values based on historical data, aiding in water resource management and operational planning for dams.




## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)




## Introduction

Accurate forecasting of dam inflow is critical for effective water resource management, flood control, hydropower generation, and agricultural planning. Unpredictable inflow patterns can lead to inefficient resource allocation, economic losses, and environmental hazards. This project presents a machine learning solution for predicting dam inflow, utilizing historical time series data. The core of this solution is a deep learning model built with TensorFlow and Keras, designed to capture complex temporal dependencies within the inflow data. By providing reliable inflow predictions, this system aims to enhance decision-making processes for dam operators and water authorities, contributing to sustainable water management practices.




## Features

-   **Data Loading and Preprocessing**: Reads historical dam inflow data from a CSV file, parses dates, and renames columns for consistency.
-   **Time Series Visualization**: Plots initial inflow data and train/test splits to provide visual insights into the data distribution and trends.
-   **Data Scaling**: Utilizes `minmax_scale` from `sklearn.preprocessing` to normalize inflow data, which is crucial for neural network performance.
-   **Windowing for Time Series**: Implements a `create_windows` function to transform sequential data into a windowed format suitable for supervised learning, considering a defined `WINDOW_SIZE` and `HORIZON`.
-   **Train/Test Split**: Divides the dataset into training and testing sets to evaluate model performance on unseen data.
-   **Sequential Keras Model**: Builds a simple yet effective feed-forward neural network using `tensorflow.keras.models.Sequential` with `Dense` layers and `relu` activation.
-   **Model Compilation**: Configures the model with the `adam` optimizer, `mse` (Mean Squared Error) loss function, and `mae` (Mean Absolute Error) as a metric.
-   **Callbacks for Training Optimization**: Incorporates `EarlyStopping` to prevent overfitting and `ReduceLROnPlateau` to dynamically adjust the learning rate during training.
-   **Training History Visualization**: Plots training and validation MAE and MSE over epochs to monitor model learning and convergence.
-   **Prediction and Inverse Scaling**: Generates predictions on the test set and inverse scales them back to the original inflow units for interpretability.
-   **Prediction Visualization**: Plots actual vs. predicted inflow values to visually assess the model's forecasting accuracy.
-   **Model Evaluation**: Calculates key regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (`r2_score`) to quantify model performance.
-   **Model Saving**: Saves the trained model in HDF5 format (`.h5`) for future use and deployment.




## Dataset

The project expects a CSV file named `lom_pangar_inflow.csv` containing historical dam inflow data. The dataset should have at least two columns: one for the `Date` and another for `inflow` values. The `Date` column is crucial for time series analysis and should be in the format `%d-%m-%Y` (day-month-year).

**Expected CSV Format:**

```csv
Date,inflow
01-01-2015,123.45
02-01-2015,125.60
03-01-2015,120.10
...
```

**Data Loading in `Inflow.py`:**

The script loads the data using `pd.read_csv` with specific parameters:

-   `"lom_pangar_inflow.csv"`: The name of the CSV file.
-   `engine='python'`: Specifies the parser engine.
-   `parse_dates=[0]`: Instructs pandas to parse the first column (index 0) as dates.
-   `index_col="Date"`: Sets the 'Date' column as the DataFrame index.
-   `date_format="%d-%m-%Y"`: Explicitly defines the date format to avoid warnings and ensure correct parsing.

Ensure your dataset adheres to this format for the script to run without modifications.




## Methodology

The dam inflow prediction methodology follows a standard time series forecasting pipeline, incorporating data preprocessing, model training, and evaluation:

1.  **Data Loading and Initial Exploration**: The `lom_pangar_inflow.csv` file is loaded into a pandas DataFrame. The 'Date' column is parsed and set as the index. The inflow data is then plotted to visualize its historical trends and patterns from 2015 to 2020.

2.  **Data Splitting**: The dataset is divided into training and testing sets. An 80/20 split is applied, where 80% of the data is used for training the model and 20% for evaluating its performance. Both the timestamps and inflow values are split accordingly.

3.  **Data Scaling**: To ensure that the neural network converges efficiently and performs optimally, the inflow data is scaled using `minmax_scale`. This transforms the data to a range between 0 and 1, preventing features with larger values from dominating the learning process.

4.  **Windowing for Time Series Forecasting**: A crucial step for time series prediction is creating a windowed dataset. The `create_windows` function generates input-output pairs where each input (`X`) consists of a sequence of `WINDOW_SIZE` (7 days) of past inflow values, and the corresponding output (`y`) is the inflow value `HORIZON` (1 day) into the future. This transforms the time series problem into a supervised learning problem.

5.  **Model Architecture Definition**: A `tensorflow.keras.models.Sequential` model is constructed. It comprises three `Dense` layers with `relu` activation, followed by a final `Dense` layer with a single output neuron, suitable for predicting a single continuous value (the dam inflow).

6.  **Model Compilation**: The model is compiled with the `adam` optimizer, which is well-suited for a wide range of deep learning tasks. The `loss` function is set to `mse` (Mean Squared Error), a common metric for regression problems, and `mae` (Mean Absolute Error) is included as an additional metric for monitoring.

7.  **Training with Callbacks**: The model is trained for a maximum of 100 epochs. To enhance training stability and prevent overfitting, two callbacks are utilized:
    -   `EarlyStopping`: Monitors the validation loss (`val_loss`) and stops training if it doesn't improve for 10 consecutive epochs, restoring the best weights observed during training.
    -   `ReduceLROnPlateau`: Reduces the learning rate by a factor of 0.2 if the validation loss plateaus for 5 epochs, with a minimum learning rate of 0.0001. This helps the model fine-tune its weights more effectively.

8.  **Prediction and Inverse Scaling**: After training, the model makes predictions on the `X_test_windowed` data. These predictions, which are in the scaled range, are then inverse-scaled back to their original units using the `inverse_minmax_scale` function. This allows for direct comparison with the actual inflow values.

9.  **Evaluation and Visualization**: The model's performance is quantitatively assessed using MAE, MSE, and R-squared on the test set. Additionally, the actual and predicted inflow values are plotted against time to visually inspect the model's forecasting accuracy and identify any discrepancies.

10. **Model Saving**: Finally, the trained model is saved as `dam_inflow_prediction_model.h5` for persistence, allowing it to be loaded and used for future predictions without retraining.




## Model Architecture

The dam inflow prediction model is a feed-forward neural network implemented using TensorFlow's Keras Sequential API. It is designed to process windowed time series data, where each input represents a sequence of past inflow values, and the output is a single predicted inflow value for a future time step.

**Sequential Model Structure:**

The model consists of four `Dense` (fully connected) layers:

-   **Input Layer**: The first `Dense` layer takes an input shape corresponding to the `WINDOW_SIZE` (7 days). This layer has 128 neurons and uses the `relu` (Rectified Linear Unit) activation function.
    -   `layers.Dense(128, activation='relu', input_shape=(WINDOW_SIZE,))`

-   **Hidden Layer 1**: A second `Dense` layer with 64 neurons, also using the `relu` activation function.
    -   `layers.Dense(64, activation='relu')`

-   **Hidden Layer 2**: A third `Dense` layer with 32 neurons, again employing the `relu` activation function.
    -   `layers.Dense(32, activation='relu')`

-   **Output Layer**: The final `Dense` layer has a single neuron and no activation function specified (implying a linear activation), which is typical for regression tasks where the output is a continuous value.
    -   `layers.Dense(1)`

**Compilation Settings:**

-   **Optimizer**: The `adam` optimizer is used. Adam (Adaptive Moment Estimation) is an efficient stochastic optimization algorithm that computes adaptive learning rates for each parameter.
-   **Loss Function**: `mse` (Mean Squared Error) is chosen as the loss function. MSE measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. It is a common and effective loss function for regression problems.
-   **Metrics**: `mae` (Mean Absolute Error) is used as an evaluation metric. MAE measures the average magnitude of the errors in a set of predictions, without considering their direction. It is a robust metric that is less sensitive to outliers compared to MSE.

**Callbacks:**

During training, two Keras callbacks are employed to optimize the learning process and prevent overfitting:

-   **`EarlyStopping`**: This callback monitors a specified metric (in this case, `val_loss` - validation loss) and stops training if the metric does not improve for a certain number of epochs (`patience=10`). It also restores the model weights from the epoch with the best value of the monitored metric.
    -   `callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`

-   **`ReduceLROnPlateau`**: This callback monitors a quantity (again, `val_loss`) and if no improvement is seen for a `patience` number of epochs (`patience=5`), the learning rate is reduced by a `factor` (0.2). A `min_lr` (0.0001) is set to prevent the learning rate from becoming too small.
    -   `callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)`

This architecture and training setup provide a robust framework for time series prediction, balancing model complexity with training efficiency and generalization capabilities.




## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/Narcisse1/DamInflowPrediction.git
    cd DamInflowPrediction
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:

    The project relies on TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, and Seaborn. You can install them using pip:

    ```bash
    pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
    ```

4.  **Prepare the Dataset**:

    Place your `lom_pangar_inflow.csv` file in the root directory of the cloned repository. Ensure it follows the format described in the [Dataset](#dataset) section.




## Usage

The `Inflow.py` script contains the entire workflow for data loading, preprocessing, model training, evaluation, and prediction. To run the project:

1.  **Ensure your dataset is in place**:

    Make sure `lom_pangar_inflow.csv` is in the same directory as `Inflow.py`.

2.  **Execute the Python script**:

    ```bash
    python Inflow.py
    ```

    The script will perform the following actions:

    -   Load and preprocess the data.
    -   Display initial plots of the inflow data.
    -   Print the lengths of the training and testing datasets.
    -   Display plots of the train/test splits.
    -   Train the Keras model, showing progress for each epoch.
    -   Display plots of the training history (MAE and MSE).
    -   Display a plot comparing actual vs. predicted inflow values.
    -   Print the Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared of the model on the test set.
    -   Save the trained model as `dam_inflow_prediction_model.h5`.

**Hyperparameters:**

You can adjust the following hyperparameters within the `Inflow.py` script:

-   `HORIZON`: Number of days into the future to predict (default: 1 for next day).
-   `WINDOW_SIZE`: Number of past days to use for prediction (default: 7 for previous week).
-   `RANDOM_SEED`: Seed for reproducibility (default: 42).
-   `epochs`: Number of training epochs (default: 100).
-   `batch_size`: Batch size for training (default: 32).

**Making New Predictions (Example):**

To use the saved model for new predictions, you would typically load the `dam_inflow_prediction_model.h5` model and provide new windowed input data. An example of how to prepare the last window of data and make a prediction for the next day is commented out at the end of `Inflow.py`:

```python
# Example of making a single prediction
# last_window = scaled_inflow[-WINDOW_SIZE:]
# last_window = last_window.reshape(1, -1)
# next_day_scaled_inflow = model.predict(last_window)
# next_day_inflow = inverse_minmax_scale(next_day_scaled_inflow, original_inflow_min, original_inflow_max)
# print(f"Predicted inflow for the next day: {next_day_inflow[0][0]:.2f}")
```

Uncomment and adapt this section if you wish to integrate the model into a larger application or make real-time predictions.




## Results

The model's performance is evaluated using several key regression metrics:

-   **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in a set of predictions, without considering their direction. A lower MAE indicates better accuracy.
-   **Mean Squared Error (MSE)**: Measures the average of the squares of the errors. It gives a relatively high weight to large errors, making it useful when large errors are particularly undesirable.
-   **R-squared (R2 Score)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R2 score closer to 1 indicates a better fit of the model to the data.

During execution, the script will print these metrics for the test set. Additionally, plots visualizing the training history (MAE and MSE over epochs) and a comparison of actual vs. predicted inflow values will be displayed. These plots provide a visual assessment of the model's learning process and its ability to forecast dam inflow accurately.

An example of the output for evaluation metrics might look like this:

```
Test Mean Absolute Error (MAE): 0.0XXX
Test Mean Squared Error (MSE): 0.0XXXX
R-squared: 0.XXXX
```

The plots generated will help in understanding the model's performance over time and its predictive capabilities.

## Author
### Narcisse Ndongkain
### ndongkainnarcisse@gmail.com


## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request. Please ensure your code adheres to the existing style and includes appropriate tests.




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




## Contact

For any questions or inquiries, please contact Narcisse1 via GitHub.


