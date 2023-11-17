import os, sys
from src.prepare_dataset import prepare_dataset

d_path = 'data/train_data.txt'
if len(sys.argv) > 1:
    d_path = sys.argv[1]

d_path = os.path.abspath(d_path)
dataset = prepare_dataset(d_path)