import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import os

# Membuat wordcloud dan menyimpannya
def create_wordcloud(text, filename):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    os.makedirs("hasil", exist_ok=True)
    path = f"hasil/{filename}"
    plt.savefig(path)
    plt.close()
    print(f"📷 WordCloud disimpan sebagai {path}")

# Visualisasi distribusi sentimen sebagai diagram batang
def plot_sentiment_distribution(df, column="Sentiment"):
    sentiment_counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color='skyblue')
    ax.set_title("Distribusi Sentimen")
    ax.set_xlabel("Label Sentimen")
    ax.set_ylabel("Jumlah")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height), ha='center', va='bottom')
    os.makedirs("hasil", exist_ok=True)
    fig.savefig("hasil/sentimen_distribution.png")
    print("📊 Diagram batang sentimen disimpan sebagai hasil/sentimen_distribution.png")

# Visualisasi frekuensi kata berdasarkan label
def plot_top_words(df, sentiment_label, filename):
    text = ' '.join(df[df['Sentiment'] == sentiment_label]['steming_data'])
    stopwords = set(STOPWORDS)
    words = [word for word in text.split() if word not in stopwords]
    word_counts = Counter(words)
    top_words = word_counts.most_common(10)
    word, count = zip(*top_words)
    colors = plt.cm.Pastel1(range(len(word)))

    plt.figure(figsize=(12, 5))
    bars = plt.bar(word, count, color=colors)
    plt.xlabel(f"Kata Sering Muncul ({sentiment_label})", fontsize=12)
    plt.ylabel("Jumlah", fontsize=12)
    plt.title(f"Frekuensi Kata - Sentimen {sentiment_label}", fontsize=14)
    plt.xticks(rotation=45)
    for bar, num in zip(bars, count):
        plt.text(bar.get_x() + bar.get_width() / 2, num + 1, str(num), ha='center')
    os.makedirs("hasil", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"hasil/{filename}")
    plt.close()
    print(f"📊 Frekuensi kata ({sentiment_label}) disimpan sebagai hasil/{filename}")
