from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

file_path = '/content/drive/MyDrive/02)2024-1학기/2_자연어처리/00_개인 프로젝트 논문 (웹툰굿즈)/데이터 및 코드/★라벨링_통합본.xlsx'
nya_df = pd.read_excel(file_path)

# 데이터 프레임 확인
nya_df.loc[302:310,:]

# 추후 구분자 문제로 발생할 tap문제를 해결하기 위해 \기호 삭제
nya_df['review'] = nya_df['review'].apply(lambda x: x.replace('\n', ' '))

# 잘 제거가 되었는지 확인해보기
nya_df.loc[302:310,:]

from sklearn.model_selection import train_test_split

train_nya, test_nya = train_test_split(nya_df, test_size=0.2, random_state=42)
print("Train Reviews : ", len(train_nya))
print("Test_Reviews : ", len(test_nya))

# Colab 환경 설정
!pip install gluonnlp pandas tqdm
!pip install mxnet
!pip install sentencepiece
!pip install transformers
!pip install torch
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# https://github.com/SKTBrain/KoBERT 의 파일들을 Colab으로 다운로드

import numpy as np
np.bool = np.bool_
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

# ★ Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from sklearn.metrics import classification_report

# GPU 사용 시
device = torch.device("cuda:0")

# ★KoBERT 토크나이저와 모델 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

# train 데이터를 .tsv 파일로 저장후 로드형식으로 변환
train_nya = train_nya.iloc[:, 2:]
train_nya.to_csv("/content/drive/MyDrive/03)DSL/4_모델링_프로젝트/지원/train_review.tsv", sep='\t', index=False)
dataset_train = nlp.data.TSVDataset("/content/drive/MyDrive/03)DSL/4_모델링_프로젝트/지원/train_review.tsv",  num_discard_samples=1)

# test 데이터를 .tsv 파일로 저장후 로드형식으로 변환
test_nya = test_nya.iloc[:, 2:]
test_nya.to_csv("/content/drive/MyDrive/03)DSL/4_모델링_프로젝트/지원/test_review.tsv", sep='\t', index=False)
dataset_test = nlp.data.TSVDataset("/content/drive/MyDrive/03)DSL/4_모델링_프로젝트/지원/test_review.tsv",  num_discard_samples=1)

train_nya.loc[302:,'review':'rating']

# ★
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        # self.labels = [print(i[label_idx]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

# ★
tok = tokenizer.tokenize
data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

#모델
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
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

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    #모델을 평가모드로 전환
    model.eval()

    # 예측 확률을 저장할 리스트 초기화
    all_probabilities = []

    # DataLoader를 통해 배치 단위로 데이터 처리
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        with torch.no_grad():
            # **모델을 통해 로짓을 계산
            logits = model(token_ids, valid_length, segment_ids)

            # **softmax를 적용하여 로짓을 확률로 변환
            probabilities = F.softmax(logits, dim=-1)
            all_probabilities.extend(probabilities.cpu().numpy())

        # 확률
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate(model, dataloader, device):
    model.eval()  # 모델을 평가 모드로 설정
    predictions, true_labels = [], []

    with torch.no_grad():  # 평가 시에는 기울기 계산을 하지 않음
        for batch in dataloader:
            # 배치를 GPU로 이동
            token_ids, valid_length, segment_ids, label = batch
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            label = label.long().to(device)

            # 평가를 위한 데이터 준비
            outputs = model(token_ids, valid_length, segment_ids)

            # 로그트 출력에서 가장 높은 확률을 가진 인덱스를 예측값으로 사용
            logits = outputs.detach().cpu().numpy()
            label_ids = label.cpu().numpy()

            # 예측값과 실제 라벨값 저장
            predictions.extend(logits.argmax(axis=-1))
            true_labels.extend(label_ids)

    # 성능 지표 계산
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

# 모델 평가
evaluation_metrics = evaluate(model, test_dataloader, device)
print(f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
print(f"Precision: {evaluation_metrics['precision']:.4f}")
print(f"Recall: {evaluation_metrics['recall']:.4f}")
print(f"F1 Score: {evaluation_metrics['f1']:.4f}")
print("\nClassification Report:\n", evaluation_metrics['classification_report'])


