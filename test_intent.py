#!/usr/bin/env python
# coding: utf-8

# In[1]:


# test_intent.py
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("intent_test.csv")  # unseen examples
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
clf = joblib.load("intent_classifier.joblib")

# Encode & predict
X = embedder.encode(df["text"].tolist(), show_progress_bar=True)
y_true = df["intent"].tolist()
y_pred = clf.predict(X)

# Report
print(classification_report(y_true, y_pred))


# In[ ]:




