import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

# Load the data
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# One-hot encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('FraudFound')
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Prepare features and target
X = data.drop(columns=['FraudFound']).values
y = data['FraudFound'].map({'Yes': 1, 'No': 0}).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define a function to create the neural network
def create_model():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))  # Input layer
    model.add(Dense(16, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Make predictions on the test set
y_test_pred = model.predict(X_test)
y_test_pred_classes = (y_test_pred > 0.5).astype(int).flatten()

# Calculate additional metrics
mse = mean_squared_error(y_test, y_test_pred_classes)
mae = mean_absolute_error(y_test, y_test_pred_classes)
rmse = np.sqrt(mse)
roc_auc = roc_auc_score(y_test, y_test_pred)

# Print the metrics
print(f'Accuracy: {test_acc:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')

#parameter tuning
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import keras_tuner as kt

# Load the data
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# One-hot encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('FraudFound')
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Prepare features and target
X = data.drop(columns=['FraudFound']).values
y = data['FraudFound'].map({'Yes': 1, 'No': 0}).values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model building function with hyperparameters
def build_model(hp):
    model = Sequential()
    
    # Choose number of hidden layers
    num_hidden_layers = hp.Int('num_hidden_layers', min_value=1, max_value=3)

    # Add input layer
    model.add(Dense(units=hp.Int('neurons_input_layer', min_value=16, max_value=64, step=16), 
                    activation=hp.Choice('activation_input_layer', values=['relu', 'tanh']), 
                    input_dim=X_train.shape[1]))
    
    # Add hidden layers
    for _ in range(num_hidden_layers - 1):
        model.add(Dense(units=hp.Int('neurons_hidden_layer', min_value=16, max_value=64, step=16), 
                        activation=hp.Choice('activation_hidden_layer', values=['relu', 'tanh'])))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Set up the tuner
tuner = kt.tuners.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='my_dir',
    project_name='fraud_detection'
)

# Perform the hyperparameter search
tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f'\nBest Test accuracy: {test_acc:.4f}')

# Make predictions on the test set
y_test_pred = best_model.predict(X_test)
y_test_pred_classes = (y_test_pred > 0.5).astype(int).flatten()

# Calculate additional metrics
mse = mean_squared_error(y_test, y_test_pred_classes)
mae = mean_absolute_error(y_test, y_test_pred_classes)
rmse = np.sqrt(mse)
roc_auc = roc_auc_score(y_test, y_test_pred)

# Print the metrics
print(f'Best Accuracy: {test_acc:.4f}')
print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
