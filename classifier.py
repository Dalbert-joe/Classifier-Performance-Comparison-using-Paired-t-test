import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(r"C:\Users\Dalbert Joe\OneDrive\Documents\Titanic-Dataset.csv")

# -----------------------------
# PREPROCESSING
# -----------------------------
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

y = df['Survived']
X = df.drop(columns=['Survived'])

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODELS (FULL NAMES)
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes (Gaussian)": GaussianNB()
}

# -----------------------------
# TRAIN & EVALUATE
# -----------------------------
results = []
cv_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    results.append([name, acc, pre, rec, f1])
    cv_scores[name] = cross_val_score(model, X, y, cv=10)

# -----------------------------
# RESULTS TABLE
# -----------------------------
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])

print("\nModel Performance\n")
print(results_df.round(4))

# -----------------------------
# GRAPH 1: ACCURACY BAR GRAPH
# -----------------------------
plt.figure()
results_df.set_index("Model")["Accuracy"].plot(kind='bar')
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# GRAPH 2: ALL METRICS BAR GRAPH
# -----------------------------
plt.figure()
results_df.set_index("Model").plot(kind='bar')
plt.title("All Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# GRAPH 3: HEATMAP
# -----------------------------
plt.figure()
names = results_df["Model"].tolist()
delta = np.zeros((len(names), len(names)))

for i in range(len(names)):
    for j in range(len(names)):
        delta[i][j] = results_df.loc[i, "Accuracy"] - results_df.loc[j, "Accuracy"]

sns.heatmap(delta, annot=True, xticklabels=names, yticklabels=names)
plt.title("Pairwise Accuracy Difference")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# PAIRED T-TEST
# -----------------------------
print("\nH0 : No Significant Difference")
print("H1 : Significant Difference\n")

ttest_data = []

for i in range(len(names)):
    for j in range(i+1, len(names)):
        t, p = ttest_rel(cv_scores[names[i]], cv_scores[names[j]])
        result = "H1" if p < 0.05 else "H0"
        ttest_data.append([names[i], names[j], round(t,4), round(p,4), result])

ttest_df = pd.DataFrame(ttest_data, columns=["Model 1", "Model 2", "t-val", "p-val", "H0/H1"])

print(ttest_df)

# -----------------------------
# BEST MODEL
# -----------------------------
best_model_name = results_df.loc[results_df["Accuracy"].idxmax(), "Model"]
best_model = models[best_model_name]

print("\nBest Model:", best_model_name)

# -----------------------------
# TEST CASES
# -----------------------------
test_cases = pd.DataFrame([
    [1, 28, 0, 1, 200.0, 0, 0, 0],
    [3, 22, 1, 0, 7.25, 1, 0, 1],
    [2, 40, 1, 2, 30.0, 1, 0, 1]
], columns=X.columns)

pred = best_model.predict(test_cases)

print("\nTest Case Results\n")

inputs = [
    "Female, 1st class, high fare",
    "Male, 3rd class, low fare",
    "Male, 2nd class, family"
]

for i in range(3):
    result = "Survived" if pred[i] == 1 else "Did Not Survive"
    print(f"Test Case {i+1}: {inputs[i]} -> {result}")

# -----------------------------
# FINAL LINE
# -----------------------------
print("\nDALBERT JOE J 311124243013")
