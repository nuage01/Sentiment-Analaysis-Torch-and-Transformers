
from string import punctuation
import importlib
import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras import callbacks
import sentencepiece as spm
from keras import backend as K
from keras.optimizers import Adam
MACHINE_LEARNING = importlib.import_module('keras_models')

# Keras==2.4.3
# tensorflow==2.4.0

class single_prediction():
    def __init__(self, sequence):
        self.review = sequence

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    #set_session( tf.compat.v1.Session(config=config))
    tf.compat.v1.Session(config=config)
    K.set_epsilon(1e-5)

    @staticmethod
    def remove_punctuation(x):
        for char in punctuation:
            x = str(x).replace(char, '')
        return x

    def predict_reviews(self):
        results = {}
        self.review_clean = self.remove_punctuation(self.review).lower()

        country = 'US'
        lang = 'EN'

        data = {'review_body': self.review, 'text_clean': self.review_clean}
        df = pd.DataFrame([data])
        df = df.dropna(subset=['review_body'])

        sp = spm.SentencePieceProcessor()
        sp.Load('../MachineLearning_processing/src/vocab.model')

        input_length, vocab_size = 256, 7500

        model = MACHINE_LEARNING.ml_model_score(vocab_size, input_length, 100)
        model.compile(optimizer=Adam(lr=1e-3),
                      loss='categorical_crossentropy', metrics=['accuracy', MACHINE_LEARNING.f1])
        model.load_weights('../MachineLearning_processing/src/score.h5')

        X = MACHINE_LEARNING.encode_docs_new_vocab(
            sp, input_length, df['review_body'])
        Y = model.predict(X, batch_size=1)
        pred_all = [1 if i[0] < 0.5 else -1 for i in Y]

        _temp = pd.DataFrame(Y)
        _temp['ml_score'] = _temp[0].apply(lambda x: 1 if x < 0.5 else -1)

        df['ml_score'] = _temp['ml_score']

        df['text_clean'] = df['text_clean'].replace(np.nan, '', regex=True)
        model = MACHINE_LEARNING.ml_model_topics(vocab_size, input_length, 100)
        model.compile(optimizer=Adam(lr=1e-3),
                      loss='binary_crossentropy', metrics=['accuracy', MACHINE_LEARNING.f1])
        model.load_weights('../MachineLearning_processing/src/topics.h5')

        X = MACHINE_LEARNING.encode_docs_new_vocab(
            sp, input_length, df['text_clean'])
        Y = model.predict(X, batch_size=1)

        targets = pickle.load(open('../MachineLearning_processing/src/targets.p', 'rb'))
        new_y = [[targets[index] if element > 0.95 else 0 for index,
                  element in enumerate(elements)] for elements in Y]

        _temp = pd.DataFrame(new_y)
        df['ml_topic'] = list(_temp[[0, 1, 2, 3, 4, 5]].values)

        df['ml_topic'] = df['ml_topic'].apply(
            lambda x: [i for i in x if i != 0])

        return df.iloc[0].to_dict()

if __name__ == "__main__":
    pass