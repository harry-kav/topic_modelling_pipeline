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

def clean_text(text: str) -> str:
    #decode any literal Python/JSON-style escapes (\n, \u2019, etc.)
    try:
        text = text.encode('utf-8').decode('unicode_escape')
    except Exception:
        #if it fails (e.g. because it's already real newlines), just carry on
        pass

    #replace newlines and tabs with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    #convert everything to ascii
    text = unidecode(text)

    #drop any remaining non-printable or non-ASCII
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    #collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'

def preprocess_text(text, min_word_len=3, lemma_strategy='spacy',nlp=None):

    stop_words = set(stopwords.words('english'))
    # Tokenize, lowercase, remove punctuation
    text = clean_text(text)
    tokens = simple_preprocess(text, deacc=True)

    # Remove stopwords
    tokens_nostop = [tok for tok in tokens if tok not in stop_words and len(tok) > min_word_len]

    if lemma_strategy == 'spacy':
        # Lemmatize with spaCy
        doc = nlp(" ".join(tokens_nostop))
        
        lemmas = [
            token.lemma_ for token in doc
            if token.lemma_.isalpha() and token.lemma_ not in stop_words
        ]
    else: #default to nltk approach if not spacy
        tagged_tokens = pos_tag(tokens_nostop)
        lemmatizer = WordNetLemmatizer()
        lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged_tokens]

    return lemmas

def main():
    file_name = 'tripadvisor_hotel_reviews.csv'
    text_col = 'Review'
    data_df = pd.read_csv(f"Data/{file_name}")
    data_df[text_col] = data_df[text_col].astype(str)
    data_df = data_df.applymap(lambda x: x.encode('unicode_escape').decode('utf-8') if isinstance(x, str) else x)

    print(data_df.head)
    print(data_df.shape)

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # run 'python -m spacy download en_core_web_sm' in terminal for this to work
    nltk.download('stopwords')

    print("Cleaning text")
    tqdm.pandas()
    data_df['clean_text'] = data_df[text_col].progress_apply(clean_text)
    print("Tokenising")
    data_df['tokens'] = data_df['clean_text'].progress_apply(preprocess_text, args=(3, 'spacy', nlp))

    data_df.to_csv(f'Data/processed_{file_name}', index=False)
    print('Preprocessing complete')

if __name__ == "__main__":
    main()

