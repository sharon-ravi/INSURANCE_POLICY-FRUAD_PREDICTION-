import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# Encode the target variable (PolicyType) using LabelEncoder
label_encoder = LabelEncoder()
data['PolicyType'] = label_encoder.fit_transform(data['PolicyType'])

# One-hot encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Separate features and target variable
X = data.drop(columns=['PolicyType']).values
y = data['PolicyType'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target variable (y_train and y_test) for multiclass classification
y_train_onehot = pd.get_dummies(y_train).values
y_test_onehot = pd.get_dummies(y_test).values

# Define a function to create the neural network for multiclass classification
def create_feedforward_nn():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X_train.shape[1]))  # Input layer
    model.add(Dense(16, activation='relu'))  # Hidden layer
    model.add(Dense(y_train_onehot.shape[1], activation='softmax'))  # Output layer for multiclass classification
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and train the feedforward neural network
model = create_feedforward_nn()
model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, verbose=0)  # Train for 50 epochs

# Make predictions on the test set
y_test_pred_proba = model.predict(X_test)

# Convert predicted probabilities to class labels
y_test_pred = np.argmax(y_test_pred_proba, axis=1)

# Evaluate the model using accuracy, MSE, MAE, and ROC AUC
accuracy = accuracy_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test_onehot, y_test_pred_proba, multi_class='ovr')

# Print evaluation metrics
print(f"Feedforward Neural Networks - Accuracy for Policy Type: {accuracy:.4f}")
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")

#parametertuning
import numpy as np
import pandas as pd
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, Input  # type: ignore
from scikeras.wrappers import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve
from keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# Clean column names
data.columns = data.columns.str.strip()

# Encode the target variable (PolicyType)
label_encoder = LabelEncoder()
data['PolicyType'] = label_encoder.fit_transform(data['PolicyType'])

# One-hot encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Prepare features and target
X = data.drop(columns=['PolicyType']).values
y = data['PolicyType'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create model (with hyperparameters for tuning)
def create_model(neurons_1=32, neurons_2=16, dropout_rate=0.2, optimizer='adam'):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Input layer
    model.add(Dense(neurons_1, activation='relu'))  # First hidden layer
    model.add(Dropout(dropout_rate))  # Dropout layer
    model.add(Dense(neurons_2, activation='relu'))  # Second hidden layer
    model.add(Dropout(dropout_rate))  # Dropout layer
    model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Wrap the model using KerasClassifier for scikit-learn compatibility
model = KerasClassifier(model=create_model, verbose=0)

# Define the hyperparameters to tune (note the prefix model__)
param_grid = {
    'model__neurons_1': [32, 64, 128],
    'model__neurons_2': [16, 32, 64],
    'model__dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30],
    'optimizer': ['adam', 'rmsprop']
}

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42)
random_search_result = random_search.fit(X_train, y_train)

# Get the best hyperparameters from the RandomizedSearchCV
print(f"Best Hyperparameters: {random_search_result.best_params_}")

# Get the best model
best_model = random_search_result.best_estimator_

# Early stopping callback to avoid overfitting during training
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with the best hyperparameters
best_model.fit(X_train, y_train, validation_split=0.2, epochs=random_search_result.best_params_['epochs'], 
               batch_size=random_search_result.best_params_['batch_size'], callbacks=[early_stopping], verbose=1)

# Make predictions on the test set
y_test_pred = best_model.predict(X_test)
if len(y_test_pred.shape) == 1:  # If predictions are 1D, treat them as class labels
    y_test_pred = y_test_pred
else:  # Otherwise, get the class with the highest probability
    y_test_pred = np.argmax(y_test_pred, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class="ovr")

# Print evaluation metrics
print(f"Final Model - Accuracy: {accuracy:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve for each class
y_test_pred_prob = best_model.predict_proba(X_test)
fpr = {}
tpr = {}

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_test_pred_prob[:, i], pos_label=i)
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC Curve')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()