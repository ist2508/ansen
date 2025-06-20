import streamlit as st
import pandas as pd
import os
from preprocessing import run_full_preprocessing
from labeling import run_labeling
from modeling import run_naive_bayes
from visualization import create_wordcloud, plot_sentiment_distribution, plot_top_words
from utils import read_csv_safely
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')

st.set_page_config(page_title="Analisis Sentimen Menggunakan Naive Bayes", layout="wide")
st.title("📊 Analisis Sentimen Menggunakan Naive Bayes")

# Tabs untuk navigasi
upload_tab, preprocess_tab, label_tab, model_tab, visual_tab = st.tabs([
    "📂 Upload Data",
    "🔄 Preprocessing",
    "🏷️ Labeling",
    "📈 Model Naive Bayes",
    "🖼️ Visualisasi"
])

# ===========================
# TAB 1: UPLOAD DATA
# ===========================
with upload_tab:
    st.subheader("📂 Unggah File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV Tweet", type="csv")
    if uploaded_file is not None:
        with open("dataMakanSiangGratis.csv", "wb") as f:
            f.write(uploaded_file.read())
        st.success("✅ File berhasil diunggah. Silakan lanjut ke tab berikutnya.")

# ===========================
# TAB 2: PREPROCESSING
# ===========================
with preprocess_tab:
    st.subheader("🔄 Tahap Preprocessing")
    if st.button("🚀 Jalankan Preprocessing"):
        with st.spinner("Sedang memproses data..."):
            df_preprocessed = run_full_preprocessing("dataMakanSiangGratis.csv")
            st.session_state.df_preprocessed = df_preprocessed
            st.success("✅ Preprocessing selesai.")

    if 'df_preprocessed' in st.session_state:
        with st.expander("📄 Lihat Hasil Preprocessing"):
            st.dataframe(st.session_state.df_preprocessed.head())

# ===========================
# TAB 3: LABELING
# ===========================
with label_tab:
    st.subheader("🏷️ Tahap Labeling Sentimen")
    if st.button("🏷️ Jalankan Labeling"):
        with st.spinner("Menentukan sentimen berdasarkan lexicon..."):
            df_labelled = run_labeling()
            st.session_state.df_labelled = df_labelled
            st.success("✅ Labeling selesai.")

    if 'df_labelled' in st.session_state:
        with st.expander("📄 Lihat Hasil Labeling"):
            st.dataframe(st.session_state.df_labelled.head())

# ===========================
# TAB 4: MODELING
# ===========================
with model_tab:
    st.subheader("📈 Naive Bayes (Multinomial)")
    if st.button("🔍 Jalankan Model Naive Bayes"):
        with st.spinner("Melatih dan mengevaluasi model..."):
            accuracy, report, conf_matrix, result_df, prior_prob, cond_pos, cond_neg, cond_net, posterior, df_posterior = run_naive_bayes()
            st.session_state.accuracy = accuracy
            st.session_state.report = report
            st.session_state.df_pred = result_df
            st.session_state.prior_prob = prior_prob
            st.session_state.cond_pos = cond_pos
            st.session_state.cond_neg = cond_neg
            st.session_state.cond_net = cond_net
            st.session_state.posterior = posterior
            st.session_state.df_posterior = df_posterior
            st.success(f"✅ Akurasi Model: {accuracy:.2f}")

    if all(k in st.session_state for k in ['prior_prob', 'cond_pos', 'cond_neg', 'cond_net', 'posterior']):
        show_details = st.checkbox("📘 Tampilkan Detail Probabilitas", value=False)
        if show_details:
            st.subheader("📊 Probabilitas Prior (Manual)")
            st.write(st.session_state.prior_prob)

            st.subheader("🔢 Probabilitas Kondisional (10 kata pertama) - Positif")
            cond_df_pos = pd.DataFrame(list(st.session_state.cond_pos.items()), columns=['Kata', 'Probabilitas']).head(10)
            st.dataframe(cond_df_pos)

            st.subheader("🔢 Probabilitas Kondisional (10 kata pertama) - Negatif")
            cond_df_neg = pd.DataFrame(list(st.session_state.cond_neg.items()), columns=['Kata', 'Probabilitas']).head(10)
            st.dataframe(cond_df_neg)

            st.subheader("🔢 Probabilitas Kondisional (10 kata pertama) - Netral")
            cond_df_net = pd.DataFrame(list(st.session_state.cond_net.items()), columns=['Kata', 'Probabilitas']).head(10)
            st.dataframe(cond_df_net)

            st.subheader("🔍 Probabilitas Posterior (Dokumen Uji Pertama)")
            st.write(st.session_state.posterior)

            st.subheader("📌 Contoh Perhitungan Posterior untuk 5 Tweet Pertama")
            st.dataframe(st.session_state.df_posterior.style.format(precision=6))

        with st.expander("📊 Laporan Evaluasi"):
            report_dict = classification_report(
                st.session_state.df_pred['Actual'],
                st.session_state.df_pred['Predicted'],
                output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            selected_cols = ['precision', 'recall', 'f1-score', 'support']
            report_df = report_df[selected_cols].round(2)
            st.dataframe(report_df)

        with st.expander("📄 Hasil Prediksi"):
            st.dataframe(st.session_state.df_pred.head())

        st.subheader("📊 Diagram Batang Prediksi Sentimen")
        sentiment_distribution = st.session_state.df_pred['Predicted'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sentiment_distribution.index, sentiment_distribution.values, color=['green', 'orange', 'red'])
        ax.set_title('Diagram Batang Hasil Analisis Sentimen Menggunakan Naive Bayes')
        ax.set_xlabel('Sentimen Prediksi')
        ax.set_ylabel('Jumlah Tweet')
        ax.set_xticks(range(len(sentiment_distribution.index)))
        ax.set_xticklabels(sentiment_distribution.index)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 5, round(yval, 2), ha='center', va='bottom')

        st.pyplot(fig)

    hasil_file = "hasil/Hasil_pred_MultinomialNB.csv"
    if os.path.exists(hasil_file):
        with open(hasil_file, "rb") as f:
            st.download_button("⬇️ Unduh Hasil Prediksi", f, file_name="hasil_sentimen.csv", mime="text/csv")

# ===========================
# TAB 5: VISUALISASI
# ===========================
with visual_tab:
    st.subheader("🖼️ Visualisasi Sentimen dan Kata")
    if st.button("📊 Buat & Tampilkan Visualisasi"):
        with st.spinner("Membuat grafik dan wordcloud..."):
            df_vis = read_csv_safely("Hasil_Labelling_Data.csv")
            if df_vis is not None:
                plot_sentiment_distribution(df_vis)
                create_wordcloud(' '.join(df_vis[df_vis['Sentiment'] == 'Negatif']['steming_data']), 'wordcloud_negatif.png')
                create_wordcloud(' '.join(df_vis[df_vis['Sentiment'] == 'Netral']['steming_data']), 'wordcloud_netral.png')
                create_wordcloud(' '.join(df_vis[df_vis['Sentiment'] == 'Positif']['steming_data']), 'wordcloud_positif.png')
                plot_top_words(df_vis, 'Negatif', 'top_words_negatif.png')
                plot_top_words(df_vis, 'Netral', 'top_words_netral.png')
                plot_top_words(df_vis, 'Positif', 'top_words_positif.png')
                st.session_state.show_visual = True

    if st.session_state.get("show_visual"):
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("hasil/wordcloud_negatif.png"):
                st.image("hasil/wordcloud_negatif.png", caption="WordCloud Negatif")
            if os.path.exists("hasil/top_words_negatif.png"):
                st.image("hasil/top_words_negatif.png", caption="Top Words Negatif")
            if os.path.exists("hasil/wordcloud_netral.png"):
                st.image("hasil/wordcloud_netral.png", caption="WordCloud Netral")
            if os.path.exists("hasil/top_words_netral.png"):
                st.image("hasil/top_words_netral.png", caption="Top Words Netral")
        with col2:
            if os.path.exists("hasil/wordcloud_positif.png"):
                st.image("hasil/wordcloud_positif.png", caption="WordCloud Positif")
            if os.path.exists("hasil/top_words_positif.png"):
                st.image("hasil/top_words_positif.png", caption="Top Words Positif")

        st.image("hasil/sentimen_distribution.png", caption="Distribusi Sentimen")
        st.image("hasil/conf_matrix_mnb.png", caption="Confusion Matrix MultinomialNB")
