import os, sys
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.prepare_dataset import prepare_dataset


def main(d):
    model = make_pipeline(
        TfidfVectorizer(),
        MultiOutputClassifier(SGDClassifier(max_iter=1000, tol=1e-3, loss='modified_huber'))
    )

    model.fit(d[0], list(zip(d[1], d[2])))
    return model


if __name__ == '__main__':
    d_path = 'data/train_data.txt'
    if len(sys.argv) > 1:
        d_path = sys.argv[1]

    d_path = os.path.abspath(d_path)
    dataset = prepare_dataset(d_path)
    m = main(dataset)

    path = os.path.dirname(d_path)
    joblib.dump(m, path + '/model.pkl')
