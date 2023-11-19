import re
import json
import joblib
import os.path
import numpy as np
from src.prepare_dataset import prepare_text
from PyDeepLX import PyDeepLX


class RequestClassifier:
    model = None
    categories = []

    def __init__(self, model_path=None):
        project_dir = os.path.dirname(__file__) + '/../'
        model_path = model_path or (project_dir + 'data/model.pkl')
        self.model = joblib.load(model_path)

    def prepare_text(self, text):
        if re.match(r'[а-яё]', text, re.IGNORECASE):
            text = PyDeepLX.translate(text)

        return prepare_text(text)

    def predict(self, text):
        text = self.prepare_text(text)
        data = self.model.predict([text])[0]

        confidence = np.max(self.model.predict_proba([text])[0])
        if confidence > 0.6:
            category = data[0]
        else:
            category = -1

        return category, data[1]
