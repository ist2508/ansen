import pandas as pd
import requests

# Load leksikon dari GitHub
positive_url = "https://raw.githubusercontent.com/fajri91/InSet/master/positive.tsv"
negative_url = "https://raw.githubusercontent.com/fajri91/InSet/master/negative.tsv"
positive_lexicon = set(pd.read_csv(positive_url, sep="\t", header=None)[0])
negative_lexicon = set(pd.read_csv(negative_url, sep="\t", header=None)[0])

# Fungsi penentuan sentimen
def determine_sentiment(text):
    if isinstance(text, str):
        positive_count = sum(1 for word in text.split() if word in positive_lexicon)
        negative_count = sum(1 for word in text.split() if word in negative_lexicon)
        score = positive_count - negative_count
        if score > 0:
            return score, "Positif"
        elif score < 0:
            return score, "Negatif"
        else:
            return score, "Netral"
    return 0, "Netral"

# Fungsi untuk melakukan labeling ke seluruh dataset
def run_labeling(preprocessed_file="Hasil_Preprocessing_Data.csv"):
    df = pd.read_csv(preprocessed_file)
    df = df[['date', 'time', 'steming_data']].dropna()
    df[['Score', 'Sentiment']] = df['steming_data'].apply(lambda x: pd.Series(determine_sentiment(x)))
    df.to_csv("Hasil_Labelling_Data.csv", index=False, encoding='utf8')
    print("✅ Labeling selesai dan disimpan sebagai Hasil_Labelling_Data.csv")
    return df
