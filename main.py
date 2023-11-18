from src.classifier import RequestClassifier

if __name__ == '__main__':
    m = RequestClassifier()

    while True:
        text = input('q: ')
        category, prior = m.predict(text)
        print(f'category: {category}, priority: {prior}')
