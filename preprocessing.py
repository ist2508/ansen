import pandas as pd
import re
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import requests
from io import BytesIO

# 1. Load dan siapkan data awal
def load_and_prepare(file_path):
    data = pd.read_csv(file_path, dtype={'created_at': str})
    data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')
    data['tanggal'] = data['created_at'].dt.date
    data['jam'] = data['created_at'].dt.time
    df = pd.DataFrame(data[['tanggal', 'jam', 'full_text']])
    df = df.rename(columns={'tanggal': 'date', 'jam': 'time', 'full_text': 'text'})
    df.drop_duplicates(subset="text", keep='first', inplace=True)
    return df

# 2. Cleaning functions
def clean_text_column(df):
    def remove_URL(tweet):
        return re.sub(r'https?://\S+|www\.\S+', '', tweet)

    def remove_html(tweet):
        return re.sub(r'<.*?>', '', tweet)

    def remove_emoji(tweet):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            u"\U0001F004-\U0001F0CF"
            u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', tweet)

    def remove_symbols(tweet):
        return re.sub(r'[^a-zA-Z0-9\s]', '', tweet)

    def remove_numbers(tweet):
        return re.sub(r'\d', '', tweet)

    def deleteHashtag(teks):
        return re.sub(r'#\w+', '', teks).strip()

    def deleteUsername(teks):
        return re.sub(r'@\w+', '', teks).strip()

    df['cleaning'] = df['text'].astype(str).apply(lambda x: remove_URL(x))
    df['cleaning'] = df['cleaning'].apply(deleteHashtag)
    df['cleaning'] = df['cleaning'].apply(deleteUsername)
    df['cleaning'] = df['cleaning'].apply(remove_html)
    df['cleaning'] = df['cleaning'].apply(remove_emoji)
    df['cleaning'] = df['cleaning'].apply(remove_symbols)
    df['cleaning'] = df['cleaning'].apply(remove_numbers)
    return df

# 3. Case folding
def apply_case_folding(df):
    df['case_folding'] = df['cleaning'].str.lower()
    return df

# 4. Normalisasi kata
def normalize_text(df):
    url = "https://github.com/analysisdatasentiment/kamus_kata_baku/raw/main/kamuskatabaku.xlsx"
    response = requests.get(url)
    kamus_data = pd.read_excel(BytesIO(response.content))
    kamus_tidak_baku_dict = dict(zip(kamus_data['tidak_baku'], kamus_data['kata_baku']))

    def replace_taboo_words(text):
        words = text.split()
        replaced_words = [kamus_tidak_baku_dict.get(w, w) for w in words]
        return ' '.join(replaced_words)

    df['normalisasi'] = df['case_folding'].apply(replace_taboo_words)
    return df

# 5. Tokenisasi, stopword removal, stemming
def tokenize_stop_stem(df):
    import nltk
    nltk.download('stopwords')
    nltk_stopwords = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    df['token'] = df['normalisasi'].apply(lambda x: x.split())
    df['stopword_removal'] = df['token'].apply(lambda x: [w for w in x if w not in nltk_stopwords])
    df['steming_data'] = df['stopword_removal'].apply(lambda x: ' '.join([stemmer.stem(w) for w in x]))
    return df

# Pipeline lengkap preprocessing
def run_full_preprocessing(file_path):
    df = load_and_prepare(file_path)
    df = clean_text_column(df)
    df = apply_case_folding(df)
    df = normalize_text(df)
    df = tokenize_stop_stem(df)
    df.dropna(inplace=True)
    df.to_csv("Hasil_Preprocessing_Data.csv", index=False, encoding='utf8')
    print("✅ Preprocessing selesai dan disimpan sebagai Hasil_Preprocessing_Data.csv")
    return df
