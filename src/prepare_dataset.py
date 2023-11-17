import json
import os.path
from Stemmer import Stemmer


def prepare_text(text: str) -> str:
    text = text.lower()

    text = text.replace(r'[^a-z]', ' ')
    text = text.replace(r'\s+[a-z]\s+', ' ')
    text = text.replace(r'\s+', ' ')

    stemmer = Stemmer('english')
    words = stemmer.stemWords(text.split())
    return ' '.join(words).strip()


def prepare_dataset(path: str) -> list:
    data, categories = [], set()
    priorities = ['standard_priority', 'medium_priority', 'high_priority']

    for line in open(path):
        line = line.strip()
        line = line.split('|')

        data.append(line)
        categories.add(line[1])

    categories = list(categories)
    for i in range(len(data)):
        data[i] = (
            prepare_text(data[i][0]),
            categories.index(data[i][1]),
            priorities.index(data[i][2])
        )

    # save data
    path = os.path.dirname(path) + '/'
    open(path + 'train_data.json', 'w').write(json.dumps(data))
    open(path + 'categories.json', 'w').write(json.dumps(categories))

    return list(map(list, zip(*data)))
