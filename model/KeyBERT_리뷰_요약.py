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
df = df[df['brand']=='마루는강쥐']
df

df1 = df[df['rating']==1]
df1_그립톡 = df1[df1['product_name']=='그립톡 (2종)'] #그립톡 (2종) 긍정 리뷰
df1_롱스티커 = df1[df1['product_name']=='롱스티커'] #롱스티커 긍정 리뷰
df1_인형키링 = df1[df1['product_name']=='미니 인형키링'] #미니 인형키링 긍정 리뷰
df1_스프링노트 = df1[df1['product_name']=='스프링노트'] #스프링노트 긍정 리뷰
df1_사각필통 = df1[df1['product_name']=='사각필통'] #사각필통 긍정 리뷰

df0 = df[df['rating']==0]
df0_그립톡 = df0[df0['product_name']=='그립톡 (2종)'] #그립톡 (2종) 부정 리뷰
df0_롱스티커 = df0[df0['product_name']=='롱스티커'] #롱스티커 부정 리뷰
df0_인형키링 = df0[df0['product_name']=='미니 인형키링'] #미니 인형키링 부정 리뷰
df0_스프링노트 = df0[df0['product_name']=='스프링노트'] #스프링노트 부정 리뷰
df0_사각필통 = df0[df0['product_name']=='사각필통'] #사각필통 부정 리뷰

#긍정 전처리
df1_그립톡["cut_special_character"]=df1_그립톡['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1_그립톡["cut_korean_consonant"]=df1_그립톡["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1_그립톡["fiterd_data"]=df1_그립톡["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0_그립톡["cut_special_character"]=df0_그립톡['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0_그립톡["cut_korean_consonant"]=df0_그립톡["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0_그립톡["fiterd_data"]=df0_그립톡["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1_그립톡['fiterd_data'])
review0 = listToString(df0_그립톡['fiterd_data'])

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

print(len(doc1), len(doc0))

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0

#긍정 전처리
df1_롱스티커["cut_special_character"]=df1_롱스티커['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1_롱스티커["cut_korean_consonant"]=df1_롱스티커["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1_롱스티커["fiterd_data"]=df1_롱스티커["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0_롱스티커["cut_special_character"]=df0_롱스티커['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0_롱스티커["cut_korean_consonant"]=df0_롱스티커["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0_롱스티커["fiterd_data"]=df0_롱스티커["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1_롱스티커['fiterd_data'])
review0 = listToString(df0_롱스티커['fiterd_data'])

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

print(len(doc1), len(doc0))

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0

#긍정 전처리
df1_인형키링["cut_special_character"]=df1_인형키링['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1_인형키링["cut_korean_consonant"]=df1_인형키링["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1_인형키링["fiterd_data"]=df1_인형키링["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0_인형키링["cut_special_character"]=df0_인형키링['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0_인형키링["cut_korean_consonant"]=df0_인형키링["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0_인형키링["fiterd_data"]=df0_인형키링["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1_인형키링['fiterd_data'])
review0 = listToString(df0_인형키링['fiterd_data'])

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

print(len(doc1), len(doc0))

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0

#긍정 전처리
df1_스프링노트["cut_special_character"]=df1_스프링노트['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1_스프링노트["cut_korean_consonant"]=df1_스프링노트["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1_스프링노트["fiterd_data"]=df1_스프링노트["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0_스프링노트["cut_special_character"]=df0_스프링노트['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0_스프링노트["cut_korean_consonant"]=df0_스프링노트["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0_스프링노트["fiterd_data"]=df0_스프링노트["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1_스프링노트['fiterd_data'])
review0 = listToString(df0_스프링노트['fiterd_data'])

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

print(len(doc1), len(doc0))

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0

#긍정 전처리
df1_사각필통["cut_special_character"]=df1_사각필통['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df1_사각필통["cut_korean_consonant"]=df1_사각필통["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df1_사각필통["fiterd_data"]=df1_사각필통["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

#부정 전처리
df0_사각필통["cut_special_character"]=df0_사각필통['review'].apply(lambda x: re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", x))
df0_사각필통["cut_korean_consonant"]=df0_사각필통["cut_special_character"].apply(lambda x: re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", x))
df0_사각필통["fiterd_data"]=df0_사각필통["cut_korean_consonant"].apply(lambda x: re.sub(r"[a-z]([A-Z])", r"-\1", x).upper())

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()

review1 = listToString(df1_사각필통['fiterd_data'])
review0 = listToString(df0_사각필통['fiterd_data'])

!pip install keybert -q

from keybert import KeyBERT
kw_model = KeyBERT()

doc1 = review1 #긍정
doc0 = review0 #부정

print(len(doc1), len(doc0))

#MMR(Maximal Margimal Relevance) 긍정
keywords_mmr_1 = kw_model.extract_keywords(doc1,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_1

#MMR(Maximal Margimal Relevance) 부정
keywords_mmr_0 = kw_model.extract_keywords(doc0,keyphrase_ngram_range=(2,4),use_mmr = True,top_n = 20,diversity = 0.3)
keywords_mmr_0

