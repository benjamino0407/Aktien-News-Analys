import streamlit as st
from transformers import pipeline
import feedparser

# Lade kostenloses Sentiment-Modell von Hugging Face
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

# Verschiedene Aktien/Themen zur Auswahl
options = ["Bitcoin", "Amazon", "Rheinmetall", "Tesla", "Nvidia"]
topic = st.selectbox("📌 Wähle ein Thema", options)

# News laden via RSS (kostenlos, keine API nötig)
def get_news(query):
    feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}+stock")
    return feed.entries[:5]

# Bewertung + Anzeige
st.title(f"📰 Nachrichten zu {topic}")

articles = get_news(topic)

for entry in articles:
    st.markdown(f"### {entry.title}")
    st.markdown(f"[🔗 Zum Artikel]({entry.link})")
    result = model(entry.title)[0]
    st.markdown(f"📊 **Stimmung**: `{result['label']}` ({round(result['score'], 2)})")
    st.markdown("---")

# Empfehlung basierend auf Durchschnittsstimmung
labels = [model(entry.title)[0]['label'] for entry in articles]
positive = labels.count("POSITIVE")
negative = labels.count("NEGATIVE")

if positive > negative:
    st.success("✅ Empfehlung: Beobachten oder Kaufen")
elif negative > positive:
    st.error("⚠️ Empfehlung: Vorsicht, lieber abwarten")
else:
    st.info("ℹ️ Empfehlung: Neutral")
