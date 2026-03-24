import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('profiles.csv')


# =========================
# 2. CLEAN DATA
# =========================

# Dropping columns that don't help prediction
df = df.drop(columns=['profile_id', 'display_name'])


# Target column
TARGET = 'label_is_fake'

X = df.drop(columns=[TARGET])
y = df[TARGET]
print(X.columns)

# =========================
# 3. FEATURE TYPES
# =========================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# =========================
# 4. PREPROCESSING
# =========================

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# =========================
# 5. MODEL
# =========================
model = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

# =========================
# 6. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

# =========================
# 7. TRAIN MODEL
# =========================
model.fit(X_train, y_train)

# =========================
# 8. EVALUATE
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))