import torch
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
from bert_classifier.dataset import CustomDataset


class BertClassifier:
    def __init__(self, model_path, tokenizer_path, data, n_classes=13, epochs=5):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.max_len = 512
        self.epochs = epochs

        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, n_classes).to(self.device)
        self.model = self.model.to(self.device)

    def preparation(self):
        self.df_train, self.df_val, self.df_test = np.split(
            self.data.sample(frac=1, random_state=42),
            [int(.85 * len(self.data)), int(.95 * len(self.data))]
        )


        self.train = CustomDataset(self.df_train, self.tokenizer, phase='train')
        self.val = CustomDataset(self.df_val, self.tokenizer, phase='train')

        self.train_dataloader = torch.utils.data.DataLoader(self.train, batch_size=4, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val, batch_size=4)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * self.epochs
        )

        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model.train()
        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(self.train_dataloader):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].squeeze(1).to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                batch_loss = self.loss_fn(output.logits, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.logits.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            total_acc_val, total_loss_val = self.eval()
            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
                | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
                | Val Loss: {total_loss_val / len(self.df_val): .3f} \
                | Val Accuracy: {total_acc_val / len(self.df_val): .3f}"
            )

            os.makedirs('/content/drive/My Drive/biv_hack/models', exist_ok=True)
            torch.save(self.model.state_dict(), f'/content/drive/My Drive/biv_hack/models/BertClassifier{epoch_num}.pt')

    def eval(self):
        self.model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in tqdm(self.val_dataloader):
                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].squeeze(1).to(self.device)
                input_id = val_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                batch_loss = self.loss_fn(output.logits, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.logits.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        return total_acc_val, total_loss_val

    def predict(self, texts):

        self.model.eval()

        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        # Отправляем данные на устройство
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        return predictions