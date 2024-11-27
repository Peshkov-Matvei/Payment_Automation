import torch
import numpy as np

CLASSES = ['SERVICE', 'NON_FOOD_GOODS', 'LOAN', 'NOT_CLASSIFIED', 'LEASING', 'FOOD_GOODS', 'BANK_SERVICE', 'TAX', 'REALE_STATE']
labels = dict(zip(CLASSES, range(len(CLASSES))))


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, phase='test'):
        self.phase = phase

        if self.phase == 'train':
            self.labels = [labels[label] for label in df['category']]
        elif self.phase == 'test':
            self.oid = [oid for oid in df['oid']]

        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
                      for text in df['text']]

    def __len__(self):
        return len(self.texts)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_oid(self, idx):
        return np.array(self.oid[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_batch_texts(idx), self.get_batch_labels(idx)
        elif self.phase == 'test':
            return self.get_batch_texts(idx), self.get_batch_oid(idx)
