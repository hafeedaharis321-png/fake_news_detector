from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
app = Flask(__name__)
models = {}
vectorizer = None
def train_if_needed():
    if not os.path.exists("models/vectorizer.pkl"):
        print("Models not found! Training now...")
        os.system("python train_models.py")
def load_all_models():
    global vectorizer, models
    train_if_needed()
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    model_files = {
        "Decision Tree": "models/decision_tree.pkl",
        "Random Forest": "models/random_forest.pkl",
        "SGD": "models/sgd.pkl",
        "KNN": "models/knn.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    print(f"Loaded {len(models)} models!")
load_all_models()
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")
    if not title and not text:
        return jsonify({"error": "Please enter news title or text!"}), 400
    combined = clean_text(title + " " + text)
    X = vectorizer.transform([combined])
    predictions = {}
    votes_fake = 0
    votes_real = 0
    for name, model in models.items():
        pred = int(model.predict(X)[0])
        label = "FAKE" if pred == 1 else "REAL"
        if pred == 1:
            votes_fake += 1
        else:
            votes_real += 1
        try:
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 1)
        except:
            confidence = None
        predictions[name] = {"label": label, "confidence": confidence}
    overall = "FAKE" if votes_fake >= votes_real else "REAL"
    return jsonify({
        "predictions": predictions,
        "overall": overall,
        "votes_fake": votes_fake,
        "votes_real": votes_real,
        "total_models": len(models)
    })
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))