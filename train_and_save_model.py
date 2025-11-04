import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load data
train_data = pd.read_csv('train_data.csv')

# Define target
target_column = 'target'
X = train_data.drop(target_column, axis=1)
y = train_data[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (StandardScaler + RandomForest with best-known params)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
])

print('Training pipeline...')
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {acc:.4f}')
print('Classification report:')
print(classification_report(y_test, y_pred))

# Save pipeline
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print('Saved pipeline to model_pipeline.pkl')
