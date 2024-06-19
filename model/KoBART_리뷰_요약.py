#터미널에서 실행
#pip install pandas torch transformers openpyxl soynlp nltk 

import re
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# Load the Excel file
df = pd.read_excel('path_to_your_excel_file.xlsx')
df = df.sort_values(by='brand')

# Filter for 긍정 (positive) and 부정 (negative) reviews for the product '그립톡 (2종)'
df1 = df[(df['rating'] == 1) & (df['product_name'] == '그립톡 (2종)')]
df0 = df[(df['rating'] == 0) & (df['product_name'] == '그립톡 (2종)')]

# Preprocessing function
def preprocess(text):
    text = re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", " ", text)
    text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)", " ", text)
    text = re.sub(r"[a-z]([A-Z])", r"-\1", text).upper()
    return text

# Apply preprocessing
df1["filtered_data"] = df1['review'].apply(preprocess)
df0["filtered_data"] = df0['review'].apply(preprocess)

# Join the reviews into a single string
review1 = " ".join(df1['filtered_data'].tolist())
review0 = " ".join(df0['filtered_data'].tolist())

# Initialize tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

# Function to summarize text
def summarize(text):
    raw_input_ids = tokenizer.encode(text)
    max_input_length = 512
    input_segments = [raw_input_ids[i:i + max_input_length] for i in range(0, len(raw_input_ids), max_input_length)]

    summary_texts = []
    for segment in input_segments:
        input_ids = [tokenizer.bos_token_id] + segment + [tokenizer.eos_token_id]
        summary_ids = model.generate(torch.tensor([input_ids]), num_beams=4, max_length=30, eos_token_id=1)
        summary_text = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summary_texts.append(summary_text)

    return " ".join(summary_texts)

# Summarize reviews
final_summary_1 = summarize(review1)
final_summary_0 = summarize(review0)

# Print summary lengths and summaries
print("긍정 리뷰 길이: ", len(review1))
print("부정 리뷰 길이: ", len(review0))
print("긍정 리뷰 요약 길이: ", len(final_summary_1))
print("부정 리뷰 요약 길이: ", len(final_summary_0))

print("긍정 리뷰 요약: ", final_summary_1)
print("부정 리뷰 요약: ", final_summary_0)
