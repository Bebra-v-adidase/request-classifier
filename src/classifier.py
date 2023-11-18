import json
import joblib
import os.path
import numpy as np


class RequestClassifier:
    model = None
    categories = []

    def __init__(self, model_path=None):
        project_dir = os.path.dirname(__file__) + '/../'
        model_path = model_path or (project_dir + 'data/model.pkl')
        self.model = joblib.load(model_path)

        d = os.path.dirname(model_path)
        with open(d + '/categories.json') as f:
            self.categories = json.loads(f.read())

    def predict(self, text):
        data = self.model.predict([text])[0]

        confidence = np.max(self.model.predict_proba([text])[0])
        if confidence > 0.6:
            category = self.categories[data[0]]
        else:
            category = 'contact_human_agent'

        return category, data[1]
