import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import pickle
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
print("Loading dataset...")
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")
fake_df['label'] = 1
real_df['label'] = 0
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Total articles:", len(df))
print("Fake news:", len(fake_df))
print("Real news:", len(real_df))
print("Preprocessing text...")
df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
df['content'] = df['content'].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
X = df['content']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Converting text to numbers...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "SGD": SGDClassifier(loss='hinge', max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric='cosine')
}
results = {}
conf_matrices = {}
for name, model in models.items():
    print("Training:", name)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    print(" Accuracy:", round(acc*100, 2), "%")
    print(" Precision:", round(prec*100, 2), "%")
    print(" Recall:", round(rec*100, 2), "%")
    print(" F1-Score:", round(f1*100, 2), "%")
    with open("models/" + name.replace(' ','_').lower() + ".pkl", "wb") as f:
        pickle.dump(model, f)
print("Saving charts...")
fig, ax = plt.subplots(figsize=(10, 6))
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_list))
width = 0.18
bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
for i, (model_name, color) in enumerate(zip(results.keys(), bar_colors)):
    vals = [results[model_name][m]*100 for m in metrics_list]
    ax.bar(x + i*width, vals, width, label=model_name, color=color, alpha=0.85)
ax.set_xlabel('Metric')
ax.set_ylabel('Score (%)')
ax.set_title('Algorithm Comparison')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics_list)
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("static/comparison_chart.png", dpi=150)
plt.close()
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (name, cm) in enumerate(conf_matrices.items()):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
    axes[i].set_title(name)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
plt.tight_layout()
plt.savefig("static/confusion_matrices.png", dpi=150)
plt.close()
names = list(results.keys())
accs = [results[n]['Accuracy']*100 for n in names]
best_model = names[accs.index(max(accs))]
colors2 = ['#4CAF50' if n == best_model else '#2196F3' for n in names]
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(names, accs, color=colors2, edgecolor='black', alpha=0.85)
ax.set_ylim(0, 115)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy Comparison')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("static/accuracy_comparison.png", dpi=150)
plt.close()
print("ALL DONE!")
print("Best Model:", best_model)
print("Now run: python app.py")