import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense, Dropout # type: ignore
import keras_tuner as kt
from keras.callbacks import EarlyStopping # type: ignore

# Load the dataset
data = pd.read_csv('/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv')

# Preprocessing
data.ffill(inplace=True)  # Fill missing values

# Assuming 'PolicyType' is the target variable
X = data.drop('PolicyType', axis=1)  # Features
y = data['PolicyType']  # Target variable

# Encode categorical features
X = pd.get_dummies(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale numerical features
X = (X - X.mean()) / X.std()  # Standardization

# Reshape X to 3D array: (samples, timesteps, features)
X = X.values.reshape(X.shape[0], 1, X.shape[1])  # Add time step of 1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optional: Use a smaller subset of training data during tuning to speed up the process
X_train_tune, _, y_train_tune, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

# Build a baseline model (before hyperparameter tuning)
baseline_model = Sequential()
baseline_model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
baseline_model.add(Dropout(0.2))
baseline_model.add(LSTM(units=64))
baseline_model.add(Dropout(0.2))
baseline_model.add(Dense(len(np.unique(y)), activation='softmax'))

# Compile the baseline model
baseline_model.compile(
    optimizer='adam',  # Default optimizer
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define EarlyStopping to prevent overtraining
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the baseline model with EarlyStopping
baseline_history = baseline_model.fit(
    X_train, y_train,
    epochs=20,  # Fewer epochs for the baseline model
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluate the baseline model
y_pred_baseline = baseline_model.predict(X_test)
y_pred_baseline_classes = np.argmax(y_pred_baseline, axis=1)

# Calculate baseline metrics
baseline_accuracy = accuracy_score(y_test, y_pred_baseline_classes)
baseline_mse = mean_squared_error(y_test, y_pred_baseline_classes)
baseline_rmse = np.sqrt(baseline_mse)
baseline_roc_auc = roc_auc_score(y_test, y_pred_baseline, multi_class='ovr')

print("\nBaseline Model Performance:")
print(f'Baseline Accuracy: {baseline_accuracy}')
print(f'Baseline RMSE: {baseline_rmse}')
print(f'Baseline MSE: {baseline_mse}')
print(f'Baseline AUC-ROC: {baseline_roc_auc}')

# Define the hyperparameter tuning function
def build_model(hp):
    model = Sequential()
    
    # Tune the number of LSTM units for the first layer
    model.add(
        LSTM(
            units=hp.Int('units', min_value=32, max_value=128, step=32),
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    
    # Tune dropout rate for the first layer
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Tune the number of LSTM units for the second layer
    model.add(
        LSTM(
            units=hp.Int('units2', min_value=32, max_value=128, step=32)
        )
    )
    
    # Tune dropout rate for the second layer
    model.add(Dropout(hp.Float('dropout_rate2', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(len(np.unique(y)), activation='softmax'))
    
    # Compile the model with a tuned optimizer
    model.compile(
        optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Set up the Keras Tuner with Random Search
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',  # Optimize for validation accuracy
    max_trials=5,  # Reduced the number of trials
    executions_per_trial=1,  # Train each model configuration once
    directory='tuner_logs',
    project_name='lstm_tuning'
)

# Run the hyperparameter search with reduced epochs and callbacks
tuner.search(X_train_tune, y_train_tune, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\nHyperparameter Tuning Results:")
print(f"The optimal number of units in the first LSTM layer is {best_hps.get('units')}")
print(f"The optimal number of units in the second LSTM layer is {best_hps.get('units2')}")
print(f"The optimal dropout rate for the first layer is {best_hps.get('dropout_rate')}")
print(f"The optimal dropout rate for the second layer is {best_hps.get('dropout_rate2')}")
print(f"The optimal optimizer is {best_hps.get('optimizer')}")

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model with the best hyperparameters and EarlyStopping
history = model.fit(
    X_train, y_train,
    epochs=50, batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Predict and evaluate metrics for the tuned model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate tuned model metrics
accuracy = accuracy_score(y_test, y_pred_classes)
mse = mean_squared_error(y_test, y_pred_classes)
rmse = np.sqrt(mse)
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

print("\nTuned Model Performance:")
print(f'Accuracy: {accuracy}')
print(f'RMSE: {rmse}')
print(f'MSE: {mse}')
print(f'AUC-ROC: {roc_auc}')