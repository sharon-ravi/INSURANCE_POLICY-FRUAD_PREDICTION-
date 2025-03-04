#final 1
import pandas as pd
import numpy as np
from keras.models import Sequential # type: ignore
from keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import roc_curve, roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt

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

# Implementing a simple version of Gradient Boosted Neural Networks
def gradient_boosted_nn(X_train, y_train, X_test, n_estimators=5):
    n_samples = len(y_train)
    y_pred = np.zeros(n_samples)
    models = []
    
    # Initial model with zero predictions (base predictions)
    for _ in range(n_estimators):
        # Calculate the residuals
        residuals = y_train - y_pred
        
        # Create a new model for the residuals
        model = create_model()
        model.fit(X_train, residuals, epochs=10, batch_size=32, verbose=0)
        
        # Update predictions
        y_pred += model.predict(X_train).flatten()  # Flatten to match y_train shape
        models.append(model)

    return models, y_pred

# Train the Gradient Boosted Neural Networks
models, y_train_pred = gradient_boosted_nn(X_train, y_train, X_test)

# Make predictions on the test set
y_test_pred = np.zeros(len(y_test))
for model in models:
    y_test_pred += model.predict(X_test).flatten()

# Convert predictions to binary (using a threshold of 0.5)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred_binary)
mse = mean_squared_error(y_test, y_test_pred_binary)
mae = mean_absolute_error(y_test, y_test_pred_binary)
roc_auc = roc_auc_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Gradient Boosted Neural Networks - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")


#parameter tuning
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.layers import Input # type: ignore


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
def create_model(neurons_1=32, neurons_2=16):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Input layer using Input
    model.add(Dense(neurons_1, activation='relu'))  # Hidden layer
    model.add(Dense(neurons_2, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_dist = {
        'neurons_1': [16, 32, 64],
        'neurons_2': [8, 16, 32],
        'epochs': [10, 20],
        'batch_size': [16, 32]
    }
    
    best_accuracy = 0
    best_params = {}
    
    # Random search over specified parameter values
    for neurons_1 in param_dist['neurons_1']:
        for neurons_2 in param_dist['neurons_2']:
            for epochs in param_dist['epochs']:
                for batch_size in param_dist['batch_size']:
                    # Create and train model
                    model = create_model(neurons_1=neurons_1, neurons_2=neurons_2)
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                    
                    # Evaluate model
                    y_train_pred = model.predict(X_train).flatten()
                    y_train_pred_binary = (y_train_pred > 0.5).astype(int)
                    accuracy = accuracy_score(y_train, y_train_pred_binary)

                    # Check if this is the best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'neurons_1': neurons_1,
                            'neurons_2': neurons_2,
                            'epochs': epochs,
                            'batch_size': batch_size
                        }
    return best_params

# Perform hyperparameter tuning
best_params = hyperparameter_tuning(X_train, y_train)
print("Best Parameters: ", best_params)

# Train the model with the best parameters
best_model = create_model(neurons_1=best_params['neurons_1'], 
                           neurons_2=best_params['neurons_2'])
best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=1)

# Make predictions on the test set
y_test_pred = best_model.predict(X_test).flatten()

# Convert predictions to binary (using a threshold of 0.5)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred_binary)
mse = mean_squared_error(y_test, y_test_pred_binary)
mae = mean_absolute_error(y_test, y_test_pred_binary)
roc_auc = roc_auc_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Final Model with Best Parameters - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.layers import Input # type: ignore
from tensorflow.keras.layers import Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import KFold

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
# Define a function to create the neural network with Dropout layers
def create_model(neurons_1=32, neurons_2=16, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Input layer
    model.add(Dense(neurons_1, activation='relu'))  # Hidden layer
    model.add(Dropout(dropout_rate))  # Dropout layer to prevent overfitting
    model.add(Dense(neurons_2, activation='relu'))  # Hidden layer
    model.add(Dropout(dropout_rate))  # Dropout layer to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Implement k-fold cross-validation to prevent overfitting
def k_fold_validation(X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_no = 1
    results = []
    
    for train_idx, val_idx in kfold.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create the model
        model = create_model(neurons_1=best_params['neurons_1'], 
                             neurons_2=best_params['neurons_2'])

        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
                  validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

        # Evaluate the model on the validation set
        y_val_pred = model.predict(X_val_fold).flatten()
        y_val_pred_binary = (y_val_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val_fold, y_val_pred_binary)
        results.append(accuracy)
        print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}")
        fold_no += 1

    avg_accuracy = np.mean(results)
    print(f"Average accuracy after k-fold cross-validation: {avg_accuracy:.4f}")
    return avg_accuracy

# Perform k-fold cross-validation to evaluate model performance
avg_accuracy = k_fold_validation(X_train, y_train)

# Train the model with the best parameters and early stopping on the entire training set
best_model = create_model(neurons_1=best_params['neurons_1'], 
                          neurons_2=best_params['neurons_2'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], 
               validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Make predictions on the test set
y_test_pred = best_model.predict(X_test).flatten()

# Convert predictions to binary (using a threshold of 0.5)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_test_pred_binary)
mse = mean_squared_error(y_test, y_test_pred_binary)
mae = mean_absolute_error(y_test, y_test_pred_binary)
roc_auc = roc_auc_score(y_test, y_test_pred)

# Print evaluation metrics
print(f"Final Model with Best Parameters - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC Curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()