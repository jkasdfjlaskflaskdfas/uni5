import joblib
import pandas as pd

model = joblib.load("major_recommender_model_v2.pkl")
encoders = joblib.load("label_encoders_v2.pkl")
features = ["favorite_subject", "location", "study_approach", "learning_env", "future_goals",
            "budget", "learn_style", "extra_activity", "scholarship"]

# Sample user input (replace with actual user answers)
user_input = {
    "favorite_subject": "Math",
    "location": "Phnom Penh",
    "study_approach": "Analytical",
    "learning_env": "Group",
    "future_goals": "Expert",
    "budget": "Medium",
    "learn_style": "Reading",
    "extra_activity": "Coding",
    "scholarship": "Needed"
}

# Encode for ML
X = pd.DataFrame([user_input], columns=features)
for col in features:
    le = encoders[col]
    X[col] = le.transform(X[col].astype(str))

# Predict probabilities
probas = model.predict_proba(X)[0]
topn_idx = probas.argsort()[-3:][::-1]
majors = encoders["target"].inverse_transform(topn_idx)
scores = probas[topn_idx]
for major, score in zip(majors, scores):
    print(f"{major} ({score:.2%})")