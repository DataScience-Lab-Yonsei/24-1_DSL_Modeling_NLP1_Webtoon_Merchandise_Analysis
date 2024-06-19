#터미널에서 실행
#pip install pandas torch transformers gluonnlp sentencepiece scikit-learn tqdm openpyxl

import re
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from transformers import BertModel, AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from kobert_tokenizer import KoBERTTokenizer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# 데이터 로드
file_path = '/content/drive/MyDrive/데이터 및 코드/라벨링_통합본.xlsx'
nya_df = pd.read_excel(file_path)

# 추후 구분자 문제로 발생할 tap문제를 해결하기 위해 \기호 삭제
nya_df['review'] = nya_df['review'].apply(lambda x: x.replace('\n', ' '))

#  train data 및 test data 분할
train_nya, test_nya = train_test_split(nya_df, test_size=0.2, random_state=42)
print("Train Reviews : ", len(train_nya))
print("Test_Reviews : ", len(test_nya))

# Save train and test datasets to .tsv files
train_nya.to_csv("train_review.tsv", sep='\t', index=False)
test_nya.to_csv("test_review.tsv", sep='\t', index=False)

# Load the datasets
dataset_train = nlp.data.TSVDataset("train_review.tsv", num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("test_review.tsv", num_discard_samples=1)

# Initialize tokenizer and model
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# Custom Dataset Class for BERT
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return self.sentences[i] + (self.labels[i], )

    def __len__(self):
        return len(self.labels)

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# Create DataLoader
data_train = BERTDataset(dataset_train, 2, 1, tokenizer.tokenize, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 2, 1, tokenizer.tokenize, vocab, max_len, True, False)

train_dataloader = DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = DataLoader(data_test, batch_size=batch_size, num_workers=5)

# Custom BERT Classifier
class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=2, dr_rate=None, params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

# Initialize model, optimizer, and loss function
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

# Function to calculate accuracy
def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

# Training and evaluation loop
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print(f"epoch {e+1} batch id {batch_id+1} loss {loss.data.cpu().numpy()} train acc {train_acc / (batch_id+1)}")
    print(f"epoch {e+1} train acc {train_acc / (batch_id+1)}")

    # Evaluation
    model.eval()
    all_probabilities = []

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        with torch.no_grad():
            logits = model(token_ids, valid_length, segment_ids)
            probabilities = F.softmax(logits, dim=-1)
            all_probabilities.extend(probabilities.cpu().numpy())
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print(f"epoch {e+1} test acc {test_acc / (batch_id+1)}")

# Function to evaluate the model
def evaluate(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            token_ids, valid_length, segment_ids, label = batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)
            outputs = model(token_ids, valid_length, segment_ids)
            logits = outputs.detach().cpu().numpy()
            label_ids = label.cpu().numpy()
            predictions.extend(logits.argmax(axis=-1))
            true_labels.extend(label_ids)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    accuracy = accuracy_score(true_labels, predictions)
    cls_report = classification_report(true_labels, predictions)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": cls_report
    }

# Evaluate the model
evaluation_metrics = evaluate(model, test_dataloader, device)
print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
print(f"Precision: {evaluation_metrics['precision']:.4f}")
print(f"Recall: {evaluation_metrics['recall']:.4f}")
print(f"F1 Score: {evaluation_metrics['f1']:.4f}")
print("\nClassification Report:\n", evaluation_metrics['classification_report'])
