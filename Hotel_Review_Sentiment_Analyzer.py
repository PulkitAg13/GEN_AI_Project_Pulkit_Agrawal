
# 🧠Hotel Review Sentiment Analyzer by Pulkit Agrawal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
import re
import nltk
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Load dataset
try:
    reviews_df = pd.read_csv("sample_reviews_data.csv")
    print("✅ Data successfully loaded.")
except:
    print("❌ Failed to load dataset. Ensure 'sample_reviews_data.csv' is in the same directory.")
    exit()

# Drop duplicates
reviews_df = reviews_df.drop_duplicates(subset='review').reset_index(drop=True)

# Sentiment + Subjectivity analysis
def analyze_text(text):
    blob = TextBlob(str(text))
    return pd.Series({
        'sentiment_score': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment_label': 'Positive' if blob.sentiment.polarity > 0.1 else 'Negative' if blob.sentiment.polarity < -0.1 else 'Neutral'
    })

sentiment_results = reviews_df['review'].apply(analyze_text)
reviews_df = pd.concat([reviews_df, sentiment_results], axis=1)

# Visual: Sentiment Distribution Pie
fig = px.pie(
    reviews_df, 
    names='sentiment_label', 
    title='Sentiment Distribution of Hotel Reviews',
    hole=0.4,
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig.show()

# Service Extraction - Enhanced
service_map = {
    'Room': ['room', 'bed', 'pillow', 'mattress', 'bathroom', 'shower', 'balcony'],
    'Staff': ['staff', 'manager', 'reception', 'housekeeping'],
    'Food': ['food', 'breakfast', 'lunch', 'dinner', 'restaurant', 'meal', 'buffet'],
    'Facilities': ['pool', 'gym', 'spa', 'wifi', 'internet', 'elevator', 'parking'],
    'Location': ['location', 'view', 'area', 'neighborhood']
}

def find_services(text):
    found = []
    tokens = re.findall(r'\b\w+\b', str(text).lower())
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    for category, keywords in service_map.items():
        if any(k in lemmas for k in keywords):
            found.append(category)
    return list(set(found))

reviews_df['service_tags'] = reviews_df['review'].apply(find_services)

# Service-wise Sentiment Summary
expanded_rows = []
for _, row in reviews_df.iterrows():
    for service in row['service_tags']:
        expanded_rows.append({
            'service': service,
            'sentiment': row['sentiment_label'],
            'score': row['sentiment_score']
        })

service_df = pd.DataFrame(expanded_rows)
summary_table = service_df.groupby(['service', 'sentiment']).size().unstack().fillna(0)
print("\n📊 Service-Wise Sentiment Table:")
print(summary_table)

# Visualize as heatmap
plt.figure(figsize=(10,6))
sns.heatmap(summary_table, annot=True, cmap='coolwarm', fmt='g')
plt.title('Sentiment by Hotel Service Category')
plt.ylabel('Service')
plt.xlabel('Sentiment')
plt.tight_layout()
plt.show()

# WordClouds
for sentiment in ['Positive', 'Negative', 'Neutral']:
    text = ' '.join(reviews_df[reviews_df['sentiment_label'] == sentiment]['review'].astype(str))
    if text:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Set2').generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{sentiment} Reviews WordCloud')
        plt.show()

# Summary Stats
total = len(reviews_df)
pos = len(reviews_df[reviews_df['sentiment_label'] == 'Positive'])
neg = len(reviews_df[reviews_df['sentiment_label'] == 'Negative'])
neu = len(reviews_df[reviews_df['sentiment_label'] == 'Neutral'])

print("\n📋 SUMMARY")
print(f"Total Reviews: {total}")
print(f"😊 Positive: {pos} ({(pos/total)*100:.1f}%)")
print(f"😞 Negative: {neg} ({(neg/total)*100:.1f}%)")
print(f"😐 Neutral: {neu} ({(neu/total)*100:.1f}%)")

# Key Takeaways
print("\n💡 TAKEAWAYS")
best = service_df.groupby('service')['score'].mean().idxmax()
worst = service_df.groupby('service')['score'].mean().idxmin()
print(f"🌟 Best rated service: {best}")
print(f"⚠️ Needs improvement: {worst}")

print("\n✅ Completed successfully.")
