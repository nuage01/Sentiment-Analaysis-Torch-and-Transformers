import numpy as np
import pandas as pd
import os, importlib
from tqdm import tqdm
import ast, pickle, spacy
from nltk.corpus import stopwords
from multiprocessing import cpu_count, Pool
import en_core_web_sm
from nltk.corpus import stopwords
import sys
import logging

logger = logging.getLogger('Journal_exemple')
logger.setLevel(logging.DEBUG)
"""full credit to 	
samer.azar@dataimpact.io
ahmad.nadar@dataimpact.io
Data Scientists at https://dataimpact.io/
"""

def get_info(retailer, country):
    
    lang, nlp_model, stop_words = 'EN', 'en_core_web_lg', set(stopwords.words('english'))
    if country in ('FR', 'BE'):
        lang, nlp_model, stop_words = 'FR', 'fr_core_news_md', set(stopwords.words('french'))
        
    global list_pos
    list_pos = ['ADJ NOUN', 'ADJ NOUN NOUN', 'ADV ADJ', 'ADV ADJ NOUN', 'ADV VERB', 'ADV VERB ADJ', 'VERB NOUN',
                'VERB ADV', 'VERB PRON PART VERB ADJ', 'ADV ADV PRON VERB', 'ADV VERB PRON', 'NOUN VERB ADJ', 
                'ADJ NOUN CCONJ NOUN', 'NOUN ADP PRON', 'VERB ADV ADJ', 'PRON VERB VERB', 'PRON VERB ADJ ADV',
                'ADV ADV ADJ', 'ADJ PUNCT ADJ NOUN NOUN', 'NOUN VERB ADV', 'ADV NOUN PRON VERB', 'ADJ CCONJ VERB ADJ', 
                'ADJ NOUN ADJ NOUN', 'ADJ ADJ NOUN']
    
    if country == 'FR':
        list_pos = ['ADJ NOUN', 'NOUN ADJ', 'NOUN ADJ NOUN', 'NOUN NOUN NOUN ADJ', 'ADV NOUN', 'NOUN ADV ADJ', 
            'NOUN ADP ADJ', 'NOUN ADP NOUN', 'ADV ADV NOUN', 'ADV ADP', 'VERB ADJ', 'ADJ ADJ', 
            'ADJ NOUN ADJ', 'ADP ADJ', 'NOUN NOUN', 'NOUN NOUN ADP NOUN', 'ADV ADJ NOUN', 'NOUN ADV NOUN',
            'NOUN ADP NOUN ADV', 'NOUN ADJ ADV ADJ', 'NOUN NOUN NOUN', 'NOUN NOUN ADP NOUN', 'ADV ADP NOUN',
            'NOUN ADP NOUN', 'ADJ NOUN NOUN NOUN', 'ADJ NOUN NOUN ADJ', 'ADJ NOUN ADP NOUN', 'ADJ NOUN ADJ NOUN', 
            'ADV ADV NOUN ADJ', 'NOUN ADP NOUN ADJ', 'ADV ADJ NOUN', 'NOUN ADJ ADV ADJ NOUN', 
            'NOUN ADJ ADP NOUN', 'NOUN ADV ADP NOUN', 'NOUN ADP VERB', 'NOUN ADV VERB', 'NOUN VERB ADV ADJ',
            'NOUN ADJ ADV ADJ', 'NOUN ADP NOUN ADJ', 'ADJ ADP NOUN', 'NOUN ADV NOUN', 'ADV ADP NOUN']
        
    return lang, nlp_model, country, retailer, stop_words, list_pos

def comment_to_opinion(t):

    opinion = text_to_nlp[t]
    opinion = generate_phrase(opinion, list_pos) 

    return (t,opinion)

def generate_phrase(text, list_pos):
    
    res = []
    for sent in text.noun_chunks:
        found = False
        op, pos = [], []
        for token in sent:
            p = token.pos_
            if p != 'PUNCT' and p != 'DET':
                to_add = str(token.lemma_)
                if to_add in opinions:
                    found = True
                op.append(to_add)
                pos.append(p)
        if found:
            op = ' '.join(op)
            pos = ' '.join(pos)
            if pos in list_pos:
                res.append(op)
    return set(res)

def read_opinions_lexicon(lang):
  
    lexico_p = pd.read_csv('./src/positive_words.txt', encoding="ISO-8859-1", sep="\n", header = None)
    lexico_p.columns = ['opinion_words']
    lexico_p = lexico_p.loc[:].reset_index(drop=True)
    set_p = set(lexico_p['opinion_words'])
    
    lexico_n = pd.read_csv('./src/negative_words.txt', encoding="ISO-8859-1", sep="\n", header = None)
    lexico_n.columns = ['opinion_words']
    lexico_n = lexico_n.loc[:].reset_index(drop=True)
    set_n = set(lexico_n['opinion_words'])

    return set_n | set_p

def process_opinions(retailer, country="UK" ,batch_size=50):
    cores = 6
    lang, nlp_model, country, retailer, stop_words, list_pos = get_info(retailer, country)
    nlp = en_core_web_sm.load()
    
    global opinions
    opinions = list(read_opinions_lexicon(lang))
    
    df =  pd.read_csv(retailer.lower() + '_cleaned.csv')
    df['review_title'] = df['review_title'].replace(np.nan, '', regex=True)
    df['review'] = df['review_body'] + ' ' + df['review_title']

    count = 0
    to_process_text = []
    for text in df['review'].unique():
        to_process_text.append(text)
        count += 1 

    global text_to_nlp

    new_text_to_opinions = {}
    text_to_nlp = {}

    _batch = 5000
    if int(count/batch_size) + 1 > 100:
        _batch = 3500

    for b in range(0, int(count/batch_size) + 1):
        _temp = to_process_text[b*batch_size:(b+1)*batch_size]
        nlp_text = [doc for doc in tqdm(nlp.pipe(_temp, batch_size=_batch))]
        text_to_nlp = {i:j for i,j in zip(_temp, nlp_text)}
        args = [elements for elements in tqdm(_temp)]

        p = Pool(cores)
        try:
            res = list(tqdm(p.imap(comment_to_opinion, args), total=len(args)))
        finally:
            p.close()
        for k,v in res:
            if v != 0:
                new_text_to_opinions[k] = v

    df['opinion'] = df['review'].apply(lambda x: list(new_text_to_opinions[x]) 
                                                                  if x in new_text_to_opinions.keys() else [''])

    df.to_csv(retailer.lower() +  '_processed_opinions.csv', index=False)

if __name__ == "__main__":
    process_opinions(sys.argv[1], 100)
    logging.info('data correctly processed')