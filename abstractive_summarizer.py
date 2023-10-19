import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import re
# from gensim.summarization import summarize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from scipy.sparse.linalg import svds
import networkx
import matplotlib.pyplot as plt
# %matplotlib inline


#paraphrasing
from transformers import *

#phrasing
# models use for this project
model_names = [
  "tuner007/pegasus_paraphrase",
  "Vamsi/T5_Paraphrase_Paws",
  "prithivida/parrot_paraphraser_on_T5", # Parrot
]




def text_regular_expression(input_text):
  GivenText = re.sub(r'\n|\r', ' ', input_text)
  GivenText = re.sub(r' +', ' ', input_text)
  GivenText = input_text.strip()
  return nltk.sent_tokenize(input_text)


def normalize_document(doc):
    #Text pre-processing using nltk
    stop_words = nltk.corpus.stopwords.words('english')
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc


# def low_rank_svd(matrix, singular_count=2):
#     u, s, vt = svds(matrix, k=singular_count)
#     return u, s, vt

def get_ranked_sentences_indices(sentences, num_sentences):
  normalize_corpus = np.vectorize(normalize_document)
  norm_sentences = normalize_corpus(sentences)

  #text representation with feature egineering
  tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
  dt_matrix = tv.fit_transform(norm_sentences)
  dt_matrix = dt_matrix.toarray()

  #Ranking the best summary
  similarity_matrix = np.matmul(dt_matrix, dt_matrix.T)

  # Get Sentence Importance Scores
  similarity_graph = networkx.from_numpy_array(similarity_matrix)
  scores = networkx.pagerank(similarity_graph)
  ranked_sentences = sorted(((score, index) for index, score in scores.items()), reverse=True)

  sen_len = len(sentences)
  if sen_len <=50:
    sum_len = sen_len
  elif sen_len > 3000:
    sum_len = 750
  elif sen_len > 50:
    sum_len = round(sen_len*0.25)
  # sum_len = round(len(sentences)/3)
  # extract the sentence no accordingto the rank
  top_sentence_indices = [ranked_sentences[index][1] for index in range(round(len(sentences)/3))]
  return top_sentence_indices


def get_paraphrased_paragraph(model, tokenizer, paragraph, num_return_sequences=5, num_beams=5):
    # Tokenize the paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)

    # Initialize an empty list to store the paraphrased sentences
    paraphrased_sentences = []

    # Paraphrase each sentence in the paragraph
    for sentence in sentences:
        inputs = tokenizer([sentence], truncation=True, padding="longest", return_tensors="pt")
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        paraphrased_sentences.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # Combine the paraphrased sentences to form a paraphrased paragraph
    paraphrased_paragraph = ' '.join(paraphrased_sentences)

    return paraphrased_paragraph

