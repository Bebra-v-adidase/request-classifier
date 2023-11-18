import json
import joblib
import os.path
import numpy as np


class RequestClassifier:
    model = None
    categories = []

    def __init__(self, model_path=None):
        model_path = model_path or (project_dir + 'data/model.pkl')
        self.model = joblib.load(model_path)

    def predict(self, text):
        data = self.model.predict([text])[0]

        confidence = np.max(self.model.predict_proba([text])[0])
        if confidence > 0.6:
            category = data[0]
        else:
            category = -1

        return category, data[1]
