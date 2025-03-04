import pandas as pd
import numpy as np
import seaborn as sns  # For advanced visualizations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from keras.models import Model # type: ignore
from keras.layers import Input, Dense # type: ignore
from keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import roc_auc_score, mean_squared_error, confusion_matrix, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# DATA PREPROCESSING
# Handle missing values (if any)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Convert 'FraudFound' from 'No'/'Yes' to 0/1
data['FraudFound'] = data['FraudFound'].map({'No': 0, 'Yes': 1})

# Verify the conversion
print("\nUnique values in 'FraudFound' after conversion:")
print(data['FraudFound'].unique())

# Define features and target
features = data.drop(columns=['FraudFound'])
target = data['FraudFound']

categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
numerical_cols = features.select_dtypes(include=[np.number]).columns.tolist()

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
X = preprocessor.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=10)

# MODEL TRAINING

# Function to create an Autoencoder model with specified hyperparameters
def create_autoencoder(hidden_layers, neurons, activation):
    input_layer = Input(shape=(X_train.shape[1],))
    x = input_layer

    # Create hidden layers
    for _ in range(hidden_layers):
        x = Dense(neurons, activation=activation)(x)
    
    decoder = Dense(X_train.shape[1], activation='sigmoid')(x)
    autoencoder = Model(input_layer, decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

# **Evaluate Overall Accuracy Before Tuning**
# Define default hyperparameters
default_hidden_layers = 1
default_neurons = 16
default_activation = 'relu'
default_epochs = 5
default_batch_size = 16

# Create and train the model with default hyperparameters
default_model = create_autoencoder(default_hidden_layers, default_neurons, default_activation)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
default_model.fit(X_train, X_train,
                  epochs=default_epochs,
                  batch_size=default_batch_size,
                  validation_split=0.2,
                  callbacks=[early_stopping],
                  verbose=0)

# Evaluate the default model
reconstructed_default = default_model.predict(X_test)
mse_values_default = np.mean(np.power(X_test - reconstructed_default, 2), axis=1)

# Set a threshold for fraud detection
threshold_default = np.percentile(mse_values_default, 95)  # Example threshold
fraudulent_claims_default = (mse_values_default > threshold_default).astype(int)

# Calculate overall accuracy
overall_accuracy_default = accuracy_score(y_test, fraudulent_claims_default)
print(f"\nOverall Accuracy Before Tuning: {overall_accuracy_default:.4f}")


# HYPERPARAMETER TUNING


best_model = None
best_params = {}
best_loss = float('inf')
tuning_count = 0  # Counter for tuning iterations

# Example hyperparameter values
hidden_layers_list = [1, 2]        
neurons_list = [8, 16, 32]         
activation_functions = ['relu', 'tanh']  
epochs_list = [5, 10]               
batch_sizes = [16, 32]              

for hidden_layers in hidden_layers_list:
    for neurons in neurons_list:
        for activation in activation_functions:
            for epoch in epochs_list:
                for batch_size in batch_sizes:
                    if tuning_count < 10:  
                        
                        autoencoder = create_autoencoder(hidden_layers, neurons, activation)
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        history = autoencoder.fit(X_train, X_train,
                                                  epochs=epoch,
                                                  batch_size=batch_size,
                                                  validation_split=0.2,
                                                  callbacks=[early_stopping],
                                                  verbose=0)

                        # Evaluate the model on the test set
                        loss = autoencoder.evaluate(X_test, X_test, verbose=0)
                        print(f"Hidden Layers: {hidden_layers}, Neurons: {neurons}, Activation: {activation}, "
                              f"Epochs: {epoch}, Batch Size: {batch_size}, Loss: {loss}")

                        # Update best model if current loss is lower
                        if loss < best_loss:
                            best_loss = loss
                            best_model = autoencoder
                            best_params = {
                                'hidden_layers': hidden_layers,
                                'neurons': neurons,
                                'activation': activation,
                                'epochs': epoch,
                                'batch_size': batch_size
                            }

                        tuning_count += 1  # Increment the counter
                    else:
                        break  # Stop if 10 combinations have been evaluated

# Output best parameters
print("\nBest Parameters:")
print(best_params)
print("Best Loss:", best_loss)

#' MODEL EVALUATION'
reconstructed = best_model.predict(X_test)
mse_values = np.mean(np.power(X_test - reconstructed, 2), axis=1)

# Set a threshold for fraud detection
threshold = np.percentile(mse_values, 95)  # Example threshold
print(f"\nReconstruction Error Threshold: {threshold}")

# Identify fraudulent claims
fraudulent_claims = (mse_values > threshold).astype(int)  
# Calculate overall accuracy after tuning
accuracy = accuracy_score(y_test, fraudulent_claims)
print(f"Accuracy: {accuracy:.4f}")

# Calculate MSE and RMSE
overall_mse = mean_squared_error(y_test, fraudulent_claims)  
overall_rmse = np.sqrt(overall_mse)  

print(f"Overall MSE: {overall_mse:.4f}")
print(f"Overall RMSE: {overall_rmse:.4f}")

# Step 1: Determine the majority class
majority_class = y_train.mode()[0]
# Step 2: Create a baseline prediction array
baseline_predictions = np.full(y_test.shape, majority_class)
# Step 3: Calculate baseline evaluation metrics
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_mse = mean_squared_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(baseline_mse)

print("\nBaseline Performance:")
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline MSE: {baseline_mse:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")

# Output results
results = pd.DataFrame({
    'Reconstruction Error': mse_values,
    'Fraudulent': fraudulent_claims,
    'Actual Fraud': y_test.values
})

# Save the results to a CSV file
results.to_csv('fraud_detection_results.csv', index=False)

# Print first few rows of the results
print("\nSample Results:")
print(results.head())

# Evaluation Metrics
auc_roc = roc_auc_score(y_test, mse_values)
print(f"\nAUC-ROC Score: {auc_roc:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, fraudulent_claims)
print("\nConfusion Matrix:")
print(conf_matrix)

# 'MODEL EVALUATION'
reconstructed = best_model.predict(X_test)
mse_values = np.mean(np.power(X_test - reconstructed, 2), axis=1)

# Set a threshold for fraud detection
threshold = np.percentile(mse_values, 95)  # Example threshold
print(f"\nReconstruction Error Threshold: {threshold}")

# Identify fraudulent claims
fraudulent_claims = (mse_values > threshold).astype(int)  

# Calculate overall accuracy after tuning
accuracy = accuracy_score(y_test, fraudulent_claims)
print(f"Accuracy: {accuracy:.4f}")

# Calculate MSE and RMSE after tuning
overall_mse_after_tuning = mean_squared_error(y_test, fraudulent_claims)  
overall_rmse_after_tuning = np.sqrt(overall_mse_after_tuning)  

# Print MSE and RMSE after tuning
print(f"Overall MSE After Tuning: {overall_mse_after_tuning:.4f}")
print(f"Overall RMSE After Tuning: {overall_rmse_after_tuning:.4f}")
