from google.colab import drive
drive.mount('/content/drive')

!pip install soynlp -q

from soynlp.word import WordExtractor
from soynlp import DoublespaceLineCorpus
from soynlp.noun import LRNounExtractor_v2
from soynlp.tokenizer import LTokenizer
import urllib.request
import re
import csv
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords

df = pd.read_excel('/content/drive/MyDrive/02)2024-1학기/2_자연어처리/00_개인 프로젝트 논문 (웹툰굿즈)/데이터 및 코드/★라벨링_통합본.xlsx')
df = df.sort_values(by='brand')
df

df1 = df[df['rating']==1]
df1 = df1[df1['product_name']=='그립톡 (2종)']
df1 #마루는 강쥐_그립톡 (2종)_긍정 리뷰

df0 = df[df['rating']==0]
df0 = df0[df0['product_name']=='그립톡 (2종)']
df0 #마루는 강쥐_그립톡 (2종)_부정 리뷰

#긍정 전처리
df1["cut_special_character"]=df1['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1["cut_korean_consonant"]=df1["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1["fiterd_data"]=df1["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0["cut_special_character"]=df0['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0["cut_korean_consonant"]=df0["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0["fiterd_data"]=df0["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

df1['fiterd_data'][0:10] #긍정

df0['fiterd_data'][0:10] #부정

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1['fiterd_data'])
review1 #긍정

review0 = listToString(df0['fiterd_data'])
review0 #부정

import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

print("긍정 리뷰 길이: ",len(review1))
print("부정 리뷰 길이: ",len(review0))

#긍정 리뷰 요약
text = review1

# 입력 텍스트를 토크나이저로 인코딩하여 토큰 ID로 변환
raw_input_ids = tokenizer.encode(text)

max_input_length = 512  # 모델이 처리할 수 있는 최대 길이

# 입력 텍스트를 최대 길이에 맞게 여러 세그먼트로 나누어 처리
input_segments = []
for i in range(0, len(raw_input_ids), max_input_length):
    input_segments.append(raw_input_ids[i:i + max_input_length])

# 각각의 입력 세그먼트에 대해 요약 생성 후 리스트에 추가
summary_texts = []
for segment in input_segments:
    input_ids = [tokenizer.bos_token_id] + segment + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=30, eos_token_id=1)
    summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    summary_texts.append(summary_text)

# 생성된 요약들을 하나로 합쳐 최종 요약 생성
final_summary_1 = " ".join(summary_texts)
final_summary_1

#긍정 리뷰 요약
text = review0

# 입력 텍스트를 토크나이저로 인코딩하여 토큰 ID로 변환
raw_input_ids = tokenizer.encode(text)

max_input_length = 512  # 모델이 처리할 수 있는 최대 길이

# 입력 텍스트를 최대 길이에 맞게 여러 세그먼트로 나누어 처리
input_segments = []
for i in range(0, len(raw_input_ids), max_input_length):
    input_segments.append(raw_input_ids[i:i + max_input_length])

# 각각의 입력 세그먼트에 대해 요약 생성 후 리스트에 추가
summary_texts = []
for segment in input_segments:
    input_ids = [tokenizer.bos_token_id] + segment + [tokenizer.eos_token_id]
    summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=30, eos_token_id=1)
    summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    summary_texts.append(summary_text)

# 생성된 요약들을 하나로 합쳐 최종 요약 생성
final_summary_0 = " ".join(summary_texts)
final_summary_0

print("긍정 리뷰 요약 길이: ",len(final_summary_1))
print("부정 리뷰 요약 길이: ",len(final_summary_0))

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

#MSS(Max Sum Similarity) 긍정
keywords_1 = keywords = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_maxsum = True,top_n = 20)
keywords_1

#MSS(Max Sum Similarity) 부정
keywords_0 = keywords = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_maxsum = True,top_n = 20)
keywords_0

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0