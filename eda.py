from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import random
import numpy as np
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split
import pandas as pd
from gensim.utils import get_random_state
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import spacy
from unidecode import unidecode
from nltk.corpus import words
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from gensim.models import Phrases
from collections import Counter
from nltk.util import ngrams
from nltk import bigrams
import ast

def get_bigrams(text):
    bi_grams = list(bigrams(text))
    return bi_grams

def plot_words(df, ngram_counts, top_n=10, filename='words'):
    #top ngrams bar chart
    top_ngrams = df.sort_values(by='count', ascending=False).head(top_n)
    plt.figure(figsize=(10,5))
    plt.bar(top_ngrams['ngram_str'], top_ngrams['count'], color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {filename}')
    plt.tight_layout()
    plt.savefig(f'Top {filename}.png')
    #plt.show(block=False)

    #bottom ngrams bar chart
    bottom_ngrams = df.sort_values(by='count').head(top_n)
    plt.figure(figsize=(10,5))
    plt.bar(bottom_ngrams['ngram_str'], bottom_ngrams['count'], color='salmon')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Least Frequent {filename}')
    plt.tight_layout()
    plt.savefig(f'Least Frequent {filename}.png')
    #plt.show(block=False)

    # wordcloud
    wordcloud_input = {' '.join(k): v for k, v in ngram_counts.items()}
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_input)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{filename} Word Cloud')
    plt.savefig(f'{filename} Word Cloud.png')
    #plt.show(block=False)

def main():
    tqdm.pandas()

    file_name = 'processed_tripadvisor_hotel_reviews.csv'
    token_col = 'tokens'

    token_df = pd.read_csv(f"Data/{file_name}")
    token_df[token_col] = token_df[token_col].apply(ast.literal_eval)

    token_df['bigrams'] = token_df[token_col].progress_apply(get_bigrams)
    token_df.to_csv(f'Data/ngrams_{file_name}', index=False)

    all_tokens = [token for row in token_df[token_col] for token in row]
    all_bigrams = [bigram for row in token_df['bigrams'] for bigram in row]

    word_counts = Counter(all_tokens)
    bigram_counts = Counter(all_bigrams)

    word_df = pd.DataFrame(word_counts.items(), columns=['ngram', 'count'])
    word_df['ngram_str'] = word_df['ngram'].progress_apply(lambda x: ' '.join(x))

    bigram_df = pd.DataFrame(bigram_counts.items(), columns=['bigram', 'count'])
    bigram_df['ngram_str'] = bigram_df['bigram'].progress_apply(lambda x: ' '.join(x))

    plot_words(word_df, word_counts, top_n=10, filename='words')
    plot_words(bigram_df, bigram_counts, top_n=10, filename='bigrams')

if __name__ == "__main__":
    main()