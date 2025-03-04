import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input # type: ignore
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, roc_auc_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

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

# Define a function to create the neural network model with Dropout layers
def create_model(neurons_1=32, neurons_2=16, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))  # Input layer
    model.add(Dense(neurons_1, activation='relu'))  # Hidden layer 1
    model.add(Dropout(dropout_rate))  # Dropout layer to prevent overfitting
    model.add(Dense(neurons_2, activation='relu'))  # Hidden layer 2
    model.add(Dropout(dropout_rate))  # Dropout layer
    model.add(Dense(len(np.unique(y)), activation='softmax'))  # Output layer for multi-class classification
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
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
        model = create_model(neurons_1=32, neurons_2=16, dropout_rate=0.3)

        # Early stopping to avoid overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, 
                  validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping], verbose=1)

        # Evaluate the model on the validation set
        y_val_pred = np.argmax(model.predict(X_val_fold), axis=1)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        results.append(accuracy)
        print(f"Fold {fold_no} - Accuracy: {accuracy:.4f}")
        fold_no += 1

    avg_accuracy = np.mean(results)
    print(f"Average accuracy after k-fold cross-validation: {avg_accuracy:.4f}")
    return avg_accuracy

# Perform k-fold cross-validation to evaluate model performance
avg_accuracy = k_fold_validation(X_train, y_train)

# Create and train the final model using the best parameters
best_model = create_model(neurons_1=32, neurons_2=16, dropout_rate=0.3)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train on the entire training set with validation split
best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

# Make predictions on the test set
y_test_pred = np.argmax(best_model.predict(X_test), axis=1)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
roc_auc = roc_auc_score(y_test, best_model.predict(X_test), multi_class="ovr")

# Print evaluation metrics
print(f"Final Model with Best Parameters - Accuracy: {accuracy:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, ROC AUC: {roc_auc:.4f}")

# Plot ROC Curve
y_test_pred_prob = best_model.predict(X_test)
fpr = {}
tpr = {}
for i in range(len(np.unique(y))):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_test_pred_prob[:, i], pos_label=i)

plt.figure(figsize=(8, 6))
for i in range(len(np.unique(y))):
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
