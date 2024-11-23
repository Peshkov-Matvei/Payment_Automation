import pandas as pd
import torch
from bert_classifier.utils import normalize_text
from bert_classifier.model import BertClassifier


model_path = 'cointegrated/rubert-tiny'
tokenizer_path = 'cointegrated/rubert-tiny'

column_name = ['number', 'date', 'sum', 'description']
df = pd.read_csv('data/payments_main.tsv', sep='\t', header=None, names=column_name)[:500]

CLASSES = ['SERVICE', 'NON_FOOD_GOODS', 'LOAN', 'NOT_CLASSIFIED', 'LEASING', 'FOOD_GOODS', 'BANK_SERVICE', 'TAX', 'REALE_STATE']
labels = dict(zip(CLASSES, range(len(CLASSES))))

df['description'] = df['description'].apply(normalize_text)

bert_tiny = BertClassifier(model_path=model_path, tokenizer_path=tokenizer_path, data=None, n_classes=len(CLASSES))
bert_tiny.model.load_state_dict(torch.load('models/final_model.pt', map_location=torch.device('cpu')))
bert_tiny.model.eval()

texts = df['description'].tolist()
predictions = bert_tiny.predict(texts)

reverse_labels = {v: k for k, v in labels.items()}

predicted_categories = [reverse_labels[pred] for pred in predictions]
print(predicted_categories)
