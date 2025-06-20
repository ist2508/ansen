import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Fungsi training dan evaluasi Multinomial Naive Bayes + manual probabilitas

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
        X_tfidf, y, X_raw, test_size=0.2, random_state=42, stratify=y
    )

    # Train MultinomialNB
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)

    # Probabilitas Prior (manual)
    sentiment_counts = y_train.value_counts()
    total_samples = len(y_train)
    prior_probabilities = sentiment_counts / total_samples

    # Probabilitas Kondisional (manual dengan Laplace smoothing)
    train_df = pd.DataFrame({'text': X_raw_train, 'label': y_train})
    all_words = " ".join(train_df['text']).split()
    features = set(all_words)

    freq_pos = {}
    freq_neg = {}
    freq_net = {}
    total_pos = total_neg = total_net = 0

    for word in features:
        freq_pos[word] = " ".join(train_df[train_df['label'] == 'Positif']['text']).split().count(word)
        freq_neg[word] = " ".join(train_df[train_df['label'] == 'Negatif']['text']).split().count(word)
        freq_net[word] = " ".join(train_df[train_df['label'] == 'Netral']['text']).split().count(word)

        total_pos += freq_pos[word]
        total_neg += freq_neg[word]
        total_net += freq_net[word]

    cond_pos = {word: (freq_pos[word] + 1) / (total_pos + len(features)) for word in features}
    cond_neg = {word: (freq_neg[word] + 1) / (total_neg + len(features)) for word in features}
    cond_net = {word: (freq_net[word] + 1) / (total_net + len(features)) for word in features}

    # Probabilitas Posterior (satu dokumen uji sebagai contoh)
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
        return {'Positif': post_pos, 'Negatif': post_neg, 'Netral': post_net}

    example_posterior = calculate_posterior(X_raw_test.iloc[0])

    # Evaluasi
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Netral', 'Positif'])
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

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

    result_df = pd.DataFrame({
        'steming_data': X_raw_test,
        'Actual': y_test,
        'Predicted': y_pred
    })
    result_df.to_csv("hasil/Hasil_pred_MultinomialNB.csv", index=False, encoding='utf8')

    return accuracy, class_report, conf_matrix, result_df, prior_probabilities, cond_pos, cond_neg, cond_net, example_posterior
