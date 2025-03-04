import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
import os
import tensorflow as tf

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('/Users/i.seviantojensima/Desktop/Sem 5/Machine Learning/ml project/final.csv')

# Check DataFrame shape
print("Initial DataFrame shape:", df.shape)

# Check for unique values in numerical columns
numerical_columns = ['Age', 'VehiclePrice', 'ClaimAmount', 'Deductible', 'DriverRating', 
                     'Days:Policy-Accident', 'Days:Policy-Claim', 'PastNumberOfClaims', 
                     'AgeOfVehicle', 'AgeOfPolicyHolder', 'NumberOfCars']

# Convert non-numeric values to NaN
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill NaNs with the median
for col in numerical_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Check for NaN values
print("Check for NaN values in numerical columns:\n", df[numerical_columns].isnull().sum())

# Preprocessing
categorical_columns = ['Make', 'AccidentArea', 'Sex', 'MaritalStatus', 'Fault', 
                       'VehicleCategory', 'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'BasePolicy']

# Encode categorical columns
if df[categorical_columns].shape[0] > 0:
    encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = encoder.fit_transform(df[categorical_columns])
else:
    encoded_columns = np.array([])

# Scale numerical columns
scaler = StandardScaler()
scaled_columns = scaler.fit_transform(df[numerical_columns])

# Combine encoded and scaled columns
if encoded_columns.size > 0 and scaled_columns.size > 0:
    X = np.concatenate([encoded_columns, scaled_columns], axis=1)
else:
    X = np.array([])

# Encode target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['PolicyType'])

# Proceed only if X is not empty
if X.size > 0:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the autoencoder
    input_dim = X_train.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, validation_split=0.2)

    # Use the encoder to get compressed features
    encoder_model = Model(inputs=input_layer, outputs=encoded)
    X_train_encoded = encoder_model.predict(X_train)
    X_test_encoded = encoder_model.predict(X_test)

    # Train classifier on encoded features before tuning
    classifier = RandomForestClassifier()
    classifier.fit(X_train_encoded, y_train)

    # Predictions before tuning
    y_pred_before = classifier.predict(X_test_encoded)

    # Evaluate accuracy before tuning
    accuracy_before = accuracy_score(y_test, y_pred_before)
    print(f"Accuracy before tuning: {accuracy_before * 100:.2f}%")

    # Calculate MSE and AUC-ROC before tuning
    mse_before = mean_squared_error(y_test, y_pred_before)
    auc_roc_before = roc_auc_score(y_test, classifier.predict_proba(X_test_encoded), multi_class='ovr')
    print(f"MSE before tuning: {mse_before:.4f}")
    print(f"AUC-ROC before tuning: {auc_roc_before:.4f}")

    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid,
                               cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_encoded, y_train)

    print(f"Best parameters from tuning: {grid_search.best_params_}")

    # Train the tuned model
    best_classifier = grid_search.best_estimator_
    best_classifier.fit(X_train_encoded, y_train)

    # Predictions after tuning
    y_pred_after = best_classifier.predict(X_test_encoded)

    # Evaluate accuracy after tuning
    accuracy_after = accuracy_score(y_test, y_pred_after)
    print(f"Accuracy after tuning: {accuracy_after * 100:.2f}%")

    # Calculate MSE and AUC-ROC after tuning
    mse_after = mean_squared_error(y_test, y_pred_after)
    auc_roc_after = roc_auc_score(y_test, best_classifier.predict_proba(X_test_encoded), multi_class='ovr')
    print(f"MSE after tuning: {mse_after:.4f}")
    print(f"AUC-ROC after tuning: {auc_roc_after:.4f}")

else:
    print("Final feature matrix X is empty; cannot proceed with model training.")
