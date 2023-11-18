import json
import joblib
import os.path


class RequestClassifier:
    model = None
    categories = []

    def __init__(self, model_path=None):
        model_path = model_path or 'data/model.pkl'
        self.model = joblib.load(model_path)

        d = os.path.dirname(model_path)
        with open(d + '/categories.json') as f:
            self.categories = json.loads(f.read())

    def predict(self, text):
        data = self.model.predict([text])[0]
        category = self.categories[data[0]]
        return category, data[1]
