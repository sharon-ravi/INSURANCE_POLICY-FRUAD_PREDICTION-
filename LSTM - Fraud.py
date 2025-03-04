import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras_tuner import RandomSearch
from tensorflow.keras.optimizers import Adam
from math import sqrt
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

# Load your dataset (adjust the path if needed)
file_path = "/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv"
data = pd.read_csv(file_path)

# Sample a smaller dataset for faster testing
data_sample = data.sample(n=1000, random_state=42)  # Adjust n as needed

# Separate features and target variable
X = data_sample.drop(columns=['FraudFound'])  # Replace 'FraudFound' with your target column name if different
y = data_sample['FraudFound'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert to binary

# Encode categorical features using LabelEncoder
X_encoded = X.copy()
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_encoded[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Split the data into training and testing sets for RFE
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize a simpler model for RFE
model = LogisticRegression(max_iter=1000)  # Ensure enough iterations

# Initialize RFE with the LogisticRegression as the estimator
n_features_to_select = 5
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)

# Fit RFE
rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_encoded.columns[rfe.support_]
print("Selected Features:")
print(selected_features)

# Visualize feature ranking
ranking = pd.DataFrame({'Feature': X_encoded.columns, 'Ranking': rfe.ranking_})
print("\nFeature Ranking:")
print(ranking.sort_values(by='Ranking'))

plt.figure(figsize=(12, 6))
plt.barh(ranking['Feature'], ranking['Ranking'], color='skyblue')
plt.xlabel('Ranking')
plt.title('Feature Ranking using RFE')
plt.gca().invert_yaxis()
plt.show()

# Use the selected features for LSTM model training
X_train_rfe = X_train[:, rfe.support_]
X_test_rfe = X_test[:, rfe.support_]

# Reshape data for LSTM [samples, time steps, features]
X_train_reshaped = X_train_rfe.reshape((X_train_rfe.shape[0], 1, X_train_rfe.shape[1]))
X_test_reshaped = X_test_rfe.reshape((X_test_rfe.shape[0], 1, X_test_rfe.shape[1]))

# Define the model-building function for Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=16),
        input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(LSTM(
        units=hp.Int('units_2', min_value=32, max_value=128, step=16),
        return_sequences=False
    ))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Set up the Keras Tuner Random Search
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='lstm_fraud_detection'
)

# Perform the search for the best hyperparameters
tuner.search(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters found
print(f"The optimal number of units in the first LSTM layer is {best_hps.get('units')}")
print(f"The optimal number of units in the second LSTM layer is {best_hps.get('units_2')}")
print(f"The optimal dropout rate for the first LSTM layer is {best_hps.get('dropout')}")
print(f"The optimal dropout rate for the second LSTM layer is {best_hps.get('dropout_2')}")
print(f"The optimal learning rate is {best_hps.get('learning_rate')}")

# Build the model with the best hyperparameters and train it
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('lstm_fraud_model.h5')

# Load the model
model = load_model('lstm_fraud_model.h5')

# Model evaluation
y_pred_prob = model.predict(X_test_reshaped)
mse = mean_squared_error(y_test, y_pred_prob)
rmse = sqrt(mse)
auc = roc_auc_score(y_test, y_pred_prob)
y_pred = (y_pred_prob > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)

# Display results
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
