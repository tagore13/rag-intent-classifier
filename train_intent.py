#!/usr/bin/env python
# coding: utf-8

# In[1]:


# train_intent.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib

# Load dataset
df = pd.read_csv("intent_data.csv")

# Encode queries
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y = df["intent"].tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Evaluation:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(clf, "intent_classifier.joblib")
print("✅ Saved model as intent_classifier.joblib")





