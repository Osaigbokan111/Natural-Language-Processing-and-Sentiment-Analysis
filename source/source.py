from nltk.stem.snowball import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
from wordcloud import WordCloud
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download(['vader_lexicon', 'stopwords',
              'punkt', 'wordnet',
              'omw-1.4'])
from nltk.probability import FreqDist

# Read the dataset into dataframe
# print the first 5 rows
senti_df = pd.read_csv('/content/Datafiniti_Hotel_Reviews.csv')
senti_df.head()

# provide concise summary of the dataset
senti_df.info()

# keep only rows with review text
senti_df = senti_df.dropna(subset=["reviews.text"])

# text cleaning function
def clean_text(text):
  tokenize_document = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)
  cleaned_document = [word.lower() for word in tokenize_document if word.lower() not in stopwords.words('english')]
  stemmed_text = [nltk.PorterStemmer().stem(word) for word in cleaned_document]
  return stemmed_text
# Apply text cleaning function
senti_df['cleaned_text'] = senti_df['reviews.text'].apply(clean_text)
senti_df.head()

# Combine all reviews into one large string
words_cloud = ' '.join(senti_df['reviews.text'].astype(str))

# Generate the word cloud
wordcloud_pos = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='Blues',
    stopwords=None  # you can add stopwords if needed
).generate(words_cloud)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Hotel Review")
plt.show()

# Instantiate VADER Analyzer
sid = SentimentIntensityAnalyzer()

# Generate Sentiment Scores
senti_df['cleaned_text_string'] = senti_df['cleaned_text'].apply(lambda x: ' '.join(x))
senti_df["polarity"] = senti_df["cleaned_text_string"].apply(lambda x: sid.polarity_scores(x))

senti_df["compound"] = senti_df["polarity"].apply(lambda score_dict: score_dict["compound"])
senti_df["neg"] = senti_df["polarity"].apply(lambda score_dict: score_dict["neg"])
senti_df["neu"] = senti_df["polarity"].apply(lambda score_dict: score_dict["neu"])
senti_df["pos"] = senti_df["polarity"].apply(lambda score_dict: score_dict["pos"])
senti_df.head()

# Sentiment scores statistics
senti_df[['compound', 'neg', 'neu', 'pos']].describe()

# Histogram distribution of compound scores
sns.histplot(senti_df["compound"])

# Histogram distribution of negative scores
sns.histplot(senti_df["neg"])

# Histogram distribution of neutral scores
sns.histplot(senti_df["neu"])

# Histogram distribution of positive scores
sns.histplot(senti_df["pos"])

# Number of negative review per hotel
negative_counts = (senti_df['compound'] <= 0).groupby(senti_df['name']).sum()
negative_counts_sorted = negative_counts.sort_values(ascending=False)
negative_counts_sorted

# Number of positive review per hotel
positive_counts = (senti_df['compound'] > 0).groupby(senti_df['name']).sum()
positive_counts_sorted = positive_counts.sort_values(ascending=False)
positive_counts_sorted

# List comprehension of word(positive and negative)
senti_df['hotel_reviews'] = senti_df['reviews.text'].apply(clean_text)
senti_df.head()
positive_review_subset = senti_df.loc[(senti_df['name'] =='The Westin Las Vegas Hotel & Spa') &
                                      (senti_df['compound'] > 0)]
positive_review_subset.head()
negative_review_subset = senti_df.loc[(senti_df['name']=='The Westin Las Vegas Hotel & Spa') &
                                      (senti_df['compound'] <= 0)]
negative_review_subset.head()

# Word Cloud for Positive Reviews
positive_words = ' '.join([' '.join(words) for words in positive_review_subset['hotel_reviews']])
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Positive Reviews (Westin Las Vegas)")
plt.show()

# Word Cloud for Negative Reviews
negative_words = ' '.join([' '.join(words) for words in negative_review_subset['hotel_reviews']])
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_words)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - Negative Reviews (Westin Las Vegas)")
plt.show()

# Flatten positive reviews into a single string of words
positive_words = ' '.join([' '.join(words) for words in positive_review_subset['hotel_reviews']])
negative_words = ' '.join([' '.join(words) for words in negative_review_subset['hotel_reviews']])

# Count word frequencies
pos_counts = Counter(positive_words.split())
neg_counts = Counter(negative_words.split())

# Display top 20 most common words for positive reviews
print("Top Positive Words:")
for word, freq in pos_counts.most_common(20):
    print(f"{word}: {freq}")

print("\nTop Negative Words:")
for word, freq in neg_counts.most_common(20):
    print(f"{word}: {freq}")

# Visualize with bar charts
def plot_word_freq(counter, title, color):
    words, freqs = zip(*counter.most_common(20))  # top 20
    plt.figure(figsize=(10,5))
    plt.bar(words, freqs, color=color)
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Frequency")
    plt.show()

plot_word_freq(pos_counts, "Top Positive Words (Westin Las Vegas)", "green")
plot_word_freq(neg_counts, "Top Negative Words (Westin Las Vegas)", "red")

