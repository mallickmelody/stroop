import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ LOAD DATASET ------------------
df = pd.read_csv("stroop_dataset.csv")

# ------------------ FEATURE ENGINEERING ------------------
df["interference"] = df["rt_incong"] - df["rt_cong"]

X = df[["age", "rt_cong", "rt_incong", "interference", "errors"]]
y = df["zone"]

# ------------------ TRAIN-TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ PIPELINE ------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Z-score from dataset
    ("model", RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        class_weight="balanced",
        random_state=42
    ))
])

# ------------------ TRAIN MODEL ------------------
pipeline.fit(X_train, y_train)

# ------------------ VALIDATION ------------------
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# ------------------ CONFUSION MATRIX ------------------
cm = confusion_matrix(y_test, y_pred, labels=["blue","green","red"])
cm_df = pd.DataFrame(cm, index=["blue","green","red"], columns=["blue","green","red"])

print("Confusion Matrix:")
print(cm_df)

# Optional: visualize confusion matrix
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# ------------------ SAVE MODEL ------------------
pickle.dump(pipeline, open("stroop_model.pkl", "wb"))
print("Model + scaler saved in stroop_model.pkl")
