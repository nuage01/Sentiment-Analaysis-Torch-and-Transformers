# single prediction with torch and transformers

# libraires install and configurations
# !pip install sentencepiece
# !pip install adaptnlp
# !pip install -q -U watermark
# !pip install -qq transformers

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext watermark
# %watermark -v -p numpy,pandas,torch,transformers
import transformers
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from textwrap import wrap
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline, AutoModelForTokenClassification
import pickle
from string import punctuation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['negative', 'positive']


#model = 'joeddav/xlm-roberta-large-xnli'
model = '../MachineLearning_processing/src/transformers_models/'

nli_model = AutoModelForSequenceClassification.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)


# Sentiment analysis pipeline
pipeline('sentiment-analysis')
SA = pipeline('sentiment-analysis',
              '../MachineLearning_processing/src/transformers_models/nlptown/')


# topics = pickle.load(
#     open('../MachineLearning_processing/src/topics.p', 'rb'), encoding='latin1')
targets = pickle.load(
    open('../MachineLearning_processing/src/targets.p', 'rb'), encoding='latin1')
targets_en = targets + ['quality']


class topics_score_pipeline():

    """Custom class that use an Natural Language Inference Model and the right encoder to
    Extract Main Topics from Reviews and scores with xlm-roberta-large model"""

    def __init__(self, nli_model, tokenizer, targets, sequence, lang='EN'):
        # self.nli_model = nli_model
        # self.tokenizer = tokenizer
        self.targets = targets
        self.sequence = sequence
        self.lang = lang


    @staticmethod
    def remove_punctuation(x):
        for char in punctuation:
            x = str(x).replace(char, '')
        return x

    
    def select_hypothesis(self):
        _hypothesis = {'FR': "Cet exemple est {}.",
                       'EN': "This exemple is {}."}

        try:
            return _hypothesis[self.lang]
        except:
            logging.error(
                'Default Language English is chosen, beceuse you didnt specify a right language [EN, FR, ES..]')
            return _hypothesis['EN']

    def topics_inference(self, sequence):
        sequence = self.remove_punctuation(sequence).lower()
        results = {}
        for label in self.targets:
            # run through model pre-trained on MNLI
            x = tokenizer.encode(sequence, self.select_hypothesis().format(label), return_tensors='pt',
                                      truncation=True)
            logits = nli_model(x)[0]

            # we throw away "neutral" (dim 1) and take the probability of
            # "entailment" (2) as the probability of the label being true
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            results[label] = (prob_label_is_true[0])

        # getting the proba value as torch item
        results = {k: v.item() for k, v in results.items()}
        # results['taste'] = results['taste'] - 0.01
        add_competition = False
        max_prob = max(results, key=results.get)
        results = {k: v for k, v in results.items(
        ) if v > 0.85 and results[max_prob] / v < 1.2}
        if max_prob == 'quality' or max_prob == 'qualité':
            if 'taste' in results.keys():
                if results['taste'] < 0.995:
                    results = list(
                        filter(lambda x: x not in 'taste', ([key for key in results.keys()])))
            elif 'goût' in results.keys():
                if results['goût'] < 0.995:
                    results = list(
                        filter(lambda x: x not in 'goût', ([key for key in results.keys()])))
        results = list(
            filter(lambda x: x not in 'quality' and x not in 'qualité', results))
        return results

    def score_inference(self, sequence):
        # get score (polarity with SA pipeline)
        _polarity = {'1': 'POSITIF', '0': 'NEGATIF'}
        res = SA(sequence)
        print(res)
        label = res[0]['label'].split(' ')[0]
        predicted_score = 1 if int(label) > 3 else 0
        return label, _polarity[str(predicted_score)]

    def live_predict(self):
        _results = {}
        topics = self.topics_inference(self.sequence)
        scores = self. score_inference(self.sequence)
        _results['topics'] = topics
        _results['text_clean'] = self.sequence
        _results['score'] = scores[1]
        return _results

def init_predict(sequence):
  extraction_object = topics_score_pipeline(nli_model, tokenizer, targets_en, sequence)
  return extraction_object.live_predict()
if __name__ == "__main__":
    pass
    # extraction_object = topics_score_pipeline(nli_model, tokenizer, SA, targets_en, 'nul cce produit')
    # data_processed = extraction_object.live_predict()
    # data_processed
