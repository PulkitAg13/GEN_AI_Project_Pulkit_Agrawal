# Hotel Review Sentiment Analyzer

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("🏨 Hotel Review Sentiment Analyzer")
print("=" * 50)

# Load the dataset
# Note: Upload your sample_reviews_data.csv file to Colab first
df = None 
dataset_loaded = False 

try:
    df = pd.read_csv('sample_reviews_data.csv')
    dataset_loaded = True # Set flag to True upon successful loading
    print(f"✅ Dataset loaded successfully!")
    print(f"📊 Total reviews: {len(df)}")
    print(f"📋 Columns: {list(df.columns)}")
    print("\n" + "="*50)
except FileNotFoundError:
    print("❌ Please upload the 'sample_reviews_data.csv' file to your Colab environment")
    print("Use the file upload button in the left sidebar")

# Only proceed with analysis if the dataset was loaded successfully
if dataset_loaded:
    # Display first few reviews
    print("🔍 Sample Reviews:")
    print(df.head(10))
    print("\n" + "="*50)

    # Check for duplicates and clean data
    print(f"🔄 Duplicate reviews found: {df.duplicated().sum()}")
    df_clean = df.drop_duplicates().reset_index(drop=True)
    print(f"✨ After removing duplicates: {len(df_clean)} reviews")
    print("\n" + "="*50)

    # Function to analyze sentiment using TextBlob
    def analyze_sentiment(text):
        """Analyze sentiment of text using TextBlob"""
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity

        if polarity > 0.1:
            return 'Positive'
        elif polarity < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def get_sentiment_score(text):
        """Get numerical sentiment score"""
        blob = TextBlob(str(text))
        return blob.sentiment.polarity

    # Apply sentiment analysis
    print("🤖 Analyzing sentiment for all reviews...")
    df_clean['sentiment'] = df_clean['review'].apply(analyze_sentiment)
    df_clean['sentiment_score'] = df_clean['review'].apply(get_sentiment_score)

    # Display sentiment distribution
    sentiment_counts = df_clean['sentiment'].value_counts()
    print("\n📈 Sentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df_clean)) * 100
        print(f"{sentiment}: {count} reviews ({percentage:.1f}%)")

    print("\n" + "="*50)

    # Visualize sentiment distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Pie chart
    colors = ['#2E8B57', '#DC143C', '#4682B4']  # Green, Red, Blue
    ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Hotel Review Sentiment Distribution', fontsize=14, fontweight='bold')

    # Bar chart
    bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=colors, alpha=0.8)
    ax2.set_title('Sentiment Count by Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Reviews')
    ax2.set_xlabel('Sentiment')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Define service categories and keywords
    service_categories = {
        'Room': ['room', 'bedroom', 'bed', 'mattress', 'pillow', 'towel', 'bathroom', 'shower', 'tv', 'minibar', 'balcony'],
        'Staff': ['staff', 'service', 'reception', 'bellboy', 'housekeeping', 'front desk', 'employee', 'manager'],
        'Food & Dining': ['food', 'breakfast', 'restaurant', 'dining', 'buffet', 'coffee', 'bar', 'drinks', 'meal'],
        'Facilities': ['wifi', 'internet', 'pool', 'swimming', 'spa', 'gym', 'elevator', 'parking', 'shuttle'],
        'Ambience': ['ambience', 'atmosphere', 'decor', 'design', 'lighting', 'view', 'location', 'noise', 'clean']
    }

    def extract_service_mentions(text, categories):
        """Extract which services are mentioned in the review"""
        text_lower = str(text).lower()
        mentioned_services = []

        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mentioned_services.append(category)
                    break

        return mentioned_services

    # Extract service mentions
    print("🔍 Extracting service-related insights...")
    df_clean['mentioned_services'] = df_clean['review'].apply(
        lambda x: extract_service_mentions(x, service_categories)
    )

    # Create service-sentiment analysis
    service_sentiment_data = []
    for idx, row in df_clean.iterrows():
        for service in row['mentioned_services']:
            service_sentiment_data.append({
                'service': service,
                'sentiment': row['sentiment'],
                'sentiment_score': row['sentiment_score'],
                'review': row['review']
            })

    service_df = pd.DataFrame(service_sentiment_data)

    if not service_df.empty:
        # Service sentiment summary
        service_summary = service_df.groupby(['service', 'sentiment']).size().unstack(fill_value=0)
        service_avg_score = service_df.groupby('service')['sentiment_score'].mean().sort_values(ascending=False)

        print("\n🏨 Service-wise Sentiment Analysis:")
        print(service_summary)
        print(f"\n⭐ Average Sentiment Scores by Service:")
        for service, score in service_avg_score.items():
            status = "😊 Positive" if score > 0.1 else "😐 Neutral" if score > -0.1 else "😞 Negative"
            print(f"{service}: {score:.3f} ({status})")

        # Visualize service sentiment
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Heatmap of service sentiment
        sns.heatmap(service_summary.T, annot=True, fmt='d', cmap='RdYlGn', ax=ax1)
        ax1.set_title('Service Sentiment Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Service Category')
        ax1.set_ylabel('Sentiment')

        # Average sentiment scores
        colors = ['green' if score > 0.1 else 'red' if score < -0.1 else 'orange'
                  for score in service_avg_score.values]
        bars = ax2.barh(service_avg_score.index, service_avg_score.values, color=colors, alpha=0.7)
        ax2.set_title('Average Sentiment Score by Service Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Average Sentiment Score')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Add score labels
        for i, (service, score) in enumerate(service_avg_score.items()):
            ax2.text(score + (0.01 if score >= 0 else -0.01), i, f'{score:.3f}',
                     va='center', ha='left' if score >= 0 else 'right', fontweight='bold')

        plt.tight_layout()
        plt.show()

    # Generate word clouds for different sentiments
    print("\n☁️ Generating Word Clouds...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sentiments = ['Positive', 'Negative', 'Neutral']
    colors = ['Greens', 'Reds', 'Blues']

    for i, sentiment in enumerate(sentiments):
        sentiment_reviews = df_clean[df_clean['sentiment'] == sentiment]['review']
        if not sentiment_reviews.empty:
            text = ' '.join(sentiment_reviews.astype(str))

            wordcloud = WordCloud(width=400, height=300,
                                 background_color='white',
                                 colormap=colors[i],
                                 max_words=100).generate(text)

            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment} Reviews Word Cloud', fontsize=14, fontweight='bold')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    # Most common positive and negative phrases
    def extract_common_phrases(reviews, sentiment_type, n=10):
        """Extract most common phrases from reviews of specific sentiment"""
        text = ' '.join(reviews.astype(str))

        # Simple bigram extraction
        words = re.findall(r'\b\w+\b', text.lower())
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]

        return Counter(bigrams).most_common(n)

    print("\n🔤 Most Common Phrases Analysis:")

    positive_reviews = df_clean[df_clean['sentiment'] == 'Positive']['review']
    negative_reviews = df_clean[df_clean['sentiment'] == 'Negative']['review']

    if not positive_reviews.empty:
        common_positive = extract_common_phrases(positive_reviews, 'Positive', 8)
        print("\n✅ Most Common POSITIVE Phrases:")
        for phrase, count in common_positive:
            print(f"  • '{phrase}': {count} times")

    if not negative_reviews.empty:
        common_negative = extract_common_phrases(negative_reviews, 'Negative', 8)
        print("\n❌ Most Common NEGATIVE Phrases:")
        for phrase, count in common_negative:
            print(f"  • '{phrase}': {count} times")

    # Summary Report Generation
    print("\n" + "="*60)
    print("📋 HOTEL REVIEW SENTIMENT ANALYSIS SUMMARY REPORT")
    print("="*60)

    total_reviews = len(df_clean)
    positive_pct = (sentiment_counts.get('Positive', 0) / total_reviews) * 100
    negative_pct = (sentiment_counts.get('Negative', 0) / total_reviews) * 100
    neutral_pct = (sentiment_counts.get('Neutral', 0) / total_reviews) * 100

    print(f"📊 Total Reviews Analyzed: {total_reviews}")
    print(f"😊 Positive Reviews: {sentiment_counts.get('Positive', 0)} ({positive_pct:.1f}%)")
    print(f"😞 Negative Reviews: {sentiment_counts.get('Negative', 0)} ({negative_pct:.1f}%)")
    print(f"😐 Neutral Reviews: {sentiment_counts.get('Neutral', 0)} ({neutral_pct:.1f}%)")

    if not service_df.empty:
        print(f"\n🏨 Service Categories Analyzed: {len(service_categories)}")
        print(f"⭐ Best Performing Service: {service_avg_score.index[0]} (Score: {service_avg_score.iloc[0]:.3f})")
        print(f"⚠️  Needs Attention: {service_avg_score.index[-1]} (Score: {service_avg_score.iloc[-1]:.3f})")

    # Key Insights
    print(f"\n💡 KEY INSIGHTS:")
    if positive_pct > 50:
        print(f"✅ Overall customer satisfaction is HIGH ({positive_pct:.1f}%)")
    else:
        print(f"⚠️  Customer satisfaction needs improvement ({positive_pct:.1f}%)")

    if not service_df.empty:
        problem_services = service_avg_score[service_avg_score < -0.1]
        if not problem_services.empty:
            print(f"🔧 Services needing immediate attention: {', '.join(problem_services.index)}")

        excellent_services = service_avg_score[service_avg_score > 0.2]
        if not excellent_services.empty:
            print(f"🌟 Services performing excellently: {', '.join(excellent_services.index)}")

    print(f"\n📈 RECOMMENDATIONS:")
    print(f"1. Focus on improving services with negative sentiment scores")
    print(f"2. Leverage positive aspects mentioned in reviews for marketing")
    print(f"3. Address specific issues mentioned in negative reviews")
    print(f"4. Monitor sentiment trends over time for continuous improvement")

    print("\n" + "="*60)
    print("✅ Analysis Complete! Use insights to improve hotel services.")
    print("="*60)
else:
    print("\nAnalysis skipped because the dataset could not be loaded.")