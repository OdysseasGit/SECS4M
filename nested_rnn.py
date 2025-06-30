"""
Surrogate Engine for Crop Simulations (SECS)
Author: Odysseas Vlachopoulos

Copyright (C) 2025 Odysseas Vlachopoulos

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

############## _______ ##############

This script implements a nested recurrent neural network (RNN) surrogate model using Keras to predict
the total weight of storage organs (TWSO) based on daily weather inputs. It reads multiple NumPy (.npy)
files containing simulated daily data from ECroPS simulations for years 1993-2023

Outputs:
  - Trained Keras model saved under “<outputdir>/<DEFINE_MODEL_NAME>”
  - Numpy arrays:
      • <DEFINE_MODEL_NAME>_predicted.npy (model predictions on X_test)
      • <DEFINE_MODEL_NAME>_ytest.npy (corresponding true TWSO values)
  - Training vs. validation loss plot saved as <DEFINE_MODEL_NAME>_trainvalloss.png

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, SimpleRNN, TimeDistributed, Dense
from tensorflow.keras.losses import Huber

# --------------------------------------------------------
# Configuration and paths
# --------------------------------------------------------

script_path = os.path.abspath(__file__)  # Absolute path of this script
# Extract version string from filename assuming format: something_v<version>.py
script_version = script_path.split('_v')[1].split('.py')[0]

# Number of training epochs and learning rate for the optimizer
num_epochs = 100
learning_rate = 0.001  

# Define the output directory for saving predictions and the trained model
# The user must replace 'ENTER PATH HERE' and 'ENTER_NUM' with actual values
resultsdir = '/path'
outputdir = os.path.join(os.sep, resultsdir, f'OUTPUT_{ENTER_NUM}')

# Define the directory that contains the ECrops simulation outputs (in .npy format)
npy_dir = '/path'

# --------------------------------------------------------
# Load and preprocess data
# --------------------------------------------------------

# List all .npy files in the specified directory, sort them, and ignore the last one
npy_files = np.sort([
    os.path.join(npy_dir, file) 
    for file in os.listdir(npy_dir) 
    if file.endswith('.npy')
])[:-1]

# Load each .npy file and concatenate along the first axis (stacking samples)
all_years = np.concatenate([np.load(file) for file in npy_files], axis=0)
print("Shape of concatenated array:", all_years.shape)
# Expected shape: (n_samples, n_days_per_year, n_total_features)

# Split into independent (features) and dependent (target) variables
# Assuming all_years has features in columns [0] and [1:] corresponding to variables
# Here: column 0 is TWSO (dependent), columns 1+ are weather features
indep_var = all_years[:, :, 1:]  # Columns 1: are weather features (TEMP_MAX, TEMP_MIN, RAINT, etc.)
dep_var = all_years[:, :, :1]    # Column 0 is the target variable (TWSO)

print('Only weather features included')
print('Independent variables - features used:', indep_var.shape)  # (n_samples, n_days, n_weather_features)
print('Dependent-target variable (TWSO):', dep_var.shape)          # (n_samples, n_days, 1)

# --------------------------------------------------------
# Add Day-of-Year (DOY) as an extra feature
# --------------------------------------------------------

# Number of samples and number of days per sample (e.g., 214 days)
n_samples = indep_var.shape[0]
n_days = indep_var.shape[1]

# Create an array [1, 2, 3, ..., n_days] to represent day-of-year
DOY = np.arange(1, n_days + 1)

# Tile the DOY array across all samples so that each sample has the same DOY sequence
DOY_expanded = np.tile(DOY, (n_samples, 1))  # Shape: (n_samples, n_days)

# Add a dimension for features so it can be concatenated (shape: (n_samples, n_days, 1))
DOY_expanded = DOY_expanded[..., np.newaxis]

# Concatenate the DOY feature to the existing independent variables along the last axis
# Resulting shape: (n_samples, n_days, n_weather_features + 1)
indep_with_DOY = np.concatenate((indep_var, DOY_expanded), axis=-1)

print(f'Independent vars with DOY shape: {indep_with_DOY.shape}')  # e.g., (135549, 214, 19)

# --------------------------------------------------------
# Trim data so that number of days is divisible by batch size
# --------------------------------------------------------

# Define how many days each batch will contain (e.g., 6 days per batch)
n_batch_days = 6

# Compute how many days to remove so that n_days is divisible by n_batch_days
days_to_remove = n_days % n_batch_days
new_n_days = n_days - days_to_remove

print(f'Days to remove: {days_to_remove}')
print(f'New number of days: {new_n_days}')

# Trim the first 'days_to_remove' days from each sample, for both features and target
indep_var_trimmed = indep_with_DOY[:, days_to_remove:, :]  # Shape: (n_samples, new_n_days, n_features)
dep_var_trimmed = dep_var[:, days_to_remove:, :]           # Shape: (n_samples, new_n_days, 1)

print(f'Trimmed X shape: {indep_var_trimmed.shape}')  # (n_samples, new_n_days, n_features)
print(f'Trimmed y shape: {dep_var_trimmed.shape}')    # (n_samples, new_n_days, 1)


# --------------------------------------------------------
# Reshape data into batches of 6 days each
# --------------------------------------------------------


def reshape_to_batches(data, batch_size):
    """
    Reshape a 3D array (samples, days, features) into a 4D array
    (samples, n_batches, batch_size, features), where n_batches = days // batch_size.
    """
    n_samples, n_days, n_features = data.shape
    n_batches = n_days // batch_size
    # New shape: (n_samples, n_batches, batch_size, n_features)
    new_shape = (n_samples, n_batches, batch_size, n_features)
    return data.reshape(new_shape)


# Apply the reshaping to independent and dependent variables
indep_data_batched = reshape_to_batches(indep_var_trimmed, n_batch_days)
dep_data_batched = reshape_to_batches(dep_var_trimmed, n_batch_days)

print(f'Reshaped independent data shape: {indep_data_batched.shape}')  # e.g., (135549, 35, 6, 19)
print(f'Reshaped dependent data shape: {dep_data_batched.shape}')      # e.g., (135549, 35, 6, 1)

# --------------------------------------------------------
# Split into training and testing sets
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    indep_data_batched, dep_data_batched, test_size=0.1
)

print(
    'X_train.shape:', X_train.shape, 
    ', X_test.shape:', X_test.shape, 
    ', y_train.shape:', y_train.shape, 
    ', y_test.shape:', y_test.shape
)

# --------------------------------------------------------
# Build the RNN model using Keras Functional API
# --------------------------------------------------------

# Extract dimensions for defining the model input shape
n_samples = indep_data_batched.shape[0]          # Total number of samples
n_outer_timesteps = indep_data_batched.shape[1]  # Number of batches per sample (e.g., 35)
n_inner_timesteps = indep_data_batched.shape[2]  # Days per batch (e.g., 6)
n_features = indep_data_batched.shape[3]         # Number of features (weather + DOY)
n_target_features = dep_data_batched.shape[3]    # Usually 1 (TWSO)
units = 128  # Number of hidden units in each LSTM layer

# Define the input layer: shape = (outer_timesteps, inner_timesteps, features)
input_layer = Input(shape=(n_outer_timesteps, n_inner_timesteps, n_features))

"""
Model architecture description:

1. TimeDistributed Inner LSTMs:
   - We wrap LSTM layers with TimeDistributed so that each sequence of 6 days 
     (inner_timesteps) is processed independently. 
   - The first TimeDistributed LSTM returns a sequence for each 6-day window.
   - The second TimeDistributed LSTM produces a fixed-length vector summarizing each 6-day window.
   - A Dropout layer follows to mitigate overfitting.

2. Outer LSTMs:
   - These LSTM layers take the sequence of 35 summarized 6-day vectors and learn
     long-term dependencies across those batches. 
   - Both outer LSTMs return sequences so we maintain a prediction for each batch.

3. TimeDistributed Dense Output:
   - Finally, a TimeDistributed Dense layer with a ReLU activation produces 
     a prediction for each 6-day batch, yielding an output shape:
     (batch_size, outer_timesteps, 1).
"""

# Inner LSTM stack applied to each 6-day segment within the 35 segments
inner_lstm = TimeDistributed(LSTM(units, return_sequences=True))(input_layer)
inner_lstm = TimeDistributed(LSTM(units, return_sequences=False))(inner_lstm)
inner_lstm = TimeDistributed(Dropout(0.3))(inner_lstm)  # Regularization

# Outer LSTM stack to process the sequence of 35 batch summaries
outer_lstm = LSTM(units, return_sequences=True)(inner_lstm)
outer_lstm = LSTM(units, return_sequences=True)(outer_lstm)

# Final output: one predicted value per 6-day batch
output_layer = TimeDistributed(Dense(1, activation='relu'))(outer_lstm)

# Build and compile the functional model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss=Huber(delta=0.5))  # Huber loss is robust to outliers

# Print a summary of the model architecture
model.summary()

# --------------------------------------------------------
# Train the model
# --------------------------------------------------------

history = model.fit(
    X_train, 
    y_train, 
    epochs=num_epochs, 
    batch_size=n_outer_timesteps,  # We treat each sample (35 batches) as one training example
    validation_data=(X_test, y_test), 
    verbose=2
)

# --------------------------------------------------------
# Evaluate the model on the test set
# --------------------------------------------------------

loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# --------------------------------------------------------
# Save the trained model and generate predictions
# --------------------------------------------------------

modelname = 'DEFINE_MODEL_NAME'  # User must define a name for saving
print(modelname)
model_filepath = os.path.join(outputdir, modelname)
model.save(model_filepath)  # Saves in TensorFlow SavedModel format

# Generate predictions on the test set
predictions = model.predict(X_test)

# Save predicted values and true test targets for later analysis/comparison
np.save(os.path.join(os.sep, outputdir, modelname + '_predicted.npy'), predictions)
np.save(os.path.join(os.sep, outputdir, modelname + '_ytest.npy'), y_test)

# --------------------------------------------------------
# Plot training and validation loss over epochs
# --------------------------------------------------------

plt.figure()
plt.style.use("ggplot")
plt.plot(history.history['loss'], label='Training data')
plt.plot(history.history['val_loss'], label='Validation data')
plt.title(modelname, fontsize=7)
plt.ylabel('Huber loss')
plt.xlabel('epoch')
plt.legend(loc="upper right")

# Save the loss curve figure to disk (300 DPI for high resolution)
plt.savefig(os.path.join(outputdir, modelname + '_trainvalloss.png'), dpi=300)
plt.close()
