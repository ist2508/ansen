import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Fungsi training dan evaluasi Multinomial Naive Bayes
def run_naive_bayes(labelled_file="Hasil_Labelling_Data.csv"):
    df = pd.read_csv(labelled_file)
    df.dropna(subset=['steming_data', 'Sentiment'], inplace=True)
    X_raw = df['steming_data']
    y = df['Sentiment']

    # TF-IDF
    tfidf = TfidfVectorizer()
    X_tfidf = tfidf.fit_transform(X_raw)

    # Split data
    X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(
        X_tfidf, y, X_raw, test_size=0.2, random_state=42, stratify=y)

    # Train MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    # Hitung Probabilitas Prior
    sentiment_counts = df['Sentiment'].value_counts()
    total_samples = len(df)
    prior_probabilities = sentiment_counts / total_samples

    # Hitung Probabilitas Kondisional Manual
    features = tfidf.get_feature_names_out()
    X_array = X_tfidf.toarray()
    feature_df = pd.DataFrame(X_array, columns=features)
    feature_df['Sentiment'] = y.values

    cond_pos = {}
    cond_neg = {}
    cond_net = {}

    for word in features:
        count_pos = feature_df[feature_df['Sentiment'] == 'Positif'][word].sum()
        count_neg = feature_df[feature_df['Sentiment'] == 'Negatif'][word].sum()
        count_net = feature_df[feature_df['Sentiment'] == 'Netral'][word].sum()

        total_pos = feature_df[feature_df['Sentiment'] == 'Positif'][features].sum().sum()
        total_neg = feature_df[feature_df['Sentiment'] == 'Negatif'][features].sum().sum()
        total_net = feature_df[feature_df['Sentiment'] == 'Netral'][features].sum().sum()

        cond_pos[word] = (count_pos + 1) / (total_pos + len(features))
        cond_neg[word] = (count_neg + 1) / (total_neg + len(features))
        cond_net[word] = (count_net + 1) / (total_net + len(features))

    # Probabilitas Posterior untuk satu dokumen
    def calculate_posterior(document):
        words = document.split()
        post_pos = prior_probabilities.get('Positif', 1)
        post_neg = prior_probabilities.get('Negatif', 1)
        post_net = prior_probabilities.get('Netral', 1)
        for word in words:
            if word in features:
                post_pos *= cond_pos[word]
                post_neg *= cond_neg[word]
                post_net *= cond_net[word]
        return {'Positif': post_pos, 'Netral': post_net, 'Negatif': post_neg}

    example_posterior = calculate_posterior(X_raw_test.iloc[0])

    # Posterior untuk beberapa tweet pertama
    posterior_list = []
    for i in range(5):
        tweet = X_raw_test.iloc[i]
        posterior = calculate_posterior(tweet)
        posterior['Tweet'] = tweet
        posterior_list.append(posterior)

    df_posterior = pd.DataFrame(posterior_list)[['Tweet', 'Positif', 'Netral', 'Negatif']]

    # Evaluasi
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Netral', 'Positif'])
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Simpan confusion matrix sebagai gambar
    os.makedirs("hasil", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negatif', 'Netral', 'Positif'],
                yticklabels=['Negatif', 'Netral', 'Positif'])
    plt.title('Confusion Matrix (MultinomialNB)')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.tight_layout()
    plt.savefig("hasil/conf_matrix_mnb.png")
    plt.close()

    # Simpan hasil prediksi
    result_df = pd.DataFrame({
        'steming_data': df.loc[y_test.index, 'steming_data'],
        'Actual': y_test,
        'Predicted': y_pred
    })
    result_df.to_csv("hasil/Hasil_pred_MultinomialNB.csv", index=False, encoding='utf8')

    return accuracy, class_report, conf_matrix, result_df, prior_probabilities, cond_pos, cond_neg, cond_net, example_posterior, df_posterior
