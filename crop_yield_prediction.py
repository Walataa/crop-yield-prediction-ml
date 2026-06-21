# COEN807 Term Project — Crop Recommendation Classification
# Olawale Adedayo Disu | P25EGCP8063
# Ahmadu Bello University, Zaria | June 2026

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train-test split (stratified, 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_enc,
    test_size=0.20,
    random_state=SEED,
    stratify=y_enc
)

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Cross-validation setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# Define models
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=SEED
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=SEED
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=SEED
    ),
}

# Train, evaluate, and report
for name, model in models.items():
    cv_scores = cross_val_score(
        model,
        X_train_sc,
        y_train,
        cv=cv,
        scoring='accuracy'
    )

    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    print(f"Model: {name}")
    print(
        f"CV Accuracy: {cv_scores.mean()*100:.2f}% +/- "
        f"{cv_scores.std()*100:.2f}%"
    )
    print(
        f"Test Accuracy: "
        f"{accuracy_score(y_test, y_pred)*100:.2f}%"
    )
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=le.classes_
        )
    )

# Feature importance (Random Forest)
rf = models['Random Forest']

feat_imp = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Feature Importance:")
print(feat_imp)