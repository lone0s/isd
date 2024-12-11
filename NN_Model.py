import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

def load_spam_dataset():
    """
    Load spam dataset with robust error handling
    """
    file_name = "spambase.data"
    path = os.path.join("datasets", "spambase", file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        try:
            url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/{file_name}"
            df = pd.read_csv(url, header=None)
            df.to_csv(path, index=False, header=False)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            raise

    return df.iloc[:, :-1].values, df.iloc[:, -1].values


X, y = load_spam_dataset()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construction du modèle
model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compilation du modèle
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Historique d'entraînement
print("Début de l'entraînement du modèle...")
training_history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Évaluation du modèle
print("\nÉvaluation du modèle sur le jeu de test :")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Prédictions
predictions = model.predict(X_test)
predictions_binary = (predictions > 0.5).astype(int)

# Matrice de confusion
print("\nMatrice de Confusion :")
cm = confusion_matrix(y_test, predictions_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de Confusion')
plt.xlabel('Prédictions')
plt.ylabel('Valeurs Réelles')
plt.show()

# Rapport de classification
print("\nRapport de Classification Détaillé :")
print(classification_report(y_test, predictions_binary))

# Visualisation de l'historique d'entraînement
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training_history.history['loss'], label='Perte Entraînement')
plt.plot(training_history.history['val_loss'], label='Perte Validation')
plt.title('Évolution de la Perte')
plt.xlabel('Époques')
plt.ylabel('Perte')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(training_history.history['accuracy'], label='Précision Entraînement')
plt.plot(training_history.history['val_accuracy'], label='Précision Validation')
plt.title('Évolution de la Précision')
plt.xlabel('Époques')
plt.ylabel('Précision')
plt.legend()

plt.tight_layout()
plt.show()

# Exemple de prédiction pour un nouvel email
print("\nExemple de prédiction :")
nouvel_email = X_test[0]  # Utilisation du premier email du jeu de test comme exemple
print(f"Mail : {X[0]}")
nouvel_email_scaled = scaler.transform(nouvel_email.reshape(1, -1))
prediction_nouvel_email = model.predict(nouvel_email_scaled)
print(f"Probabilité d'être un spam : {prediction_nouvel_email[0][0]:.4f}")
print(f"Est un spam : {prediction_nouvel_email[0][0] > 0.5}")