from pprint import pprint
from adaptnlp import EasySequenceClassifier
from string import punctuation
from pandas import DataFrame
from termcolor import colored
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from textwrap import wrap
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import rc
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from transformers import CamembertModel, CamembertTokenizer
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import transformers
from pandas import DataFrame
from string import punctuation
from adaptnlp import EasySequenceClassifier
import logging


logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger(
    'allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

%reload_ext watermark
%watermark - v - p numpy, pandas, torch, transformers
%matplotlib inline


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['negative', 'positive']

PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

BATCH_SIZE = 32
MAX_LEN = 280


"""
import Data from GCP (Dataimpact's data)

from google.colab import auth
auth.authenticate_user()

# https://cloud.google.com/resource-manager/docs/creating-managing-projects
project_id = 'dataimpact-rd'
!gcloud config set project {project_id}


# Download the file from a given Google Cloud Storage bucket.
!gsutil cp gs://di_data_sas/EN/US/Amazon/Data/periode_11/amazon_ml_opinions_topics.csv /tmp/amazon_ml_opinions_topics.csv
  
# Print the result to make sure the transfer worked.
!head -n 5 /tmp/amazon_ml_opinions_topics.csv

!gsutil cp gs://di_data_sas/EN/US/Walmart/Data/2020_periode_1/walmart_ml_opinions.csv /tmp/walmart_ml_opinions_topics.csv
!gsutil cp  gs://di_data_sas/EN/US/Target/Data/2020_periode_1/target_ml_opinions.csv /tmp/target_ml_opinions_topics.csv
!gsutil cp gs://di_data_sas/EN/UK/Asda/Data/2020_periode_1/asda_ml_opinions.csv /tmp/asda_ml_opinions_topics.csv
!gsutil cp gs://di_data_sas/EN/UK/Morrisons/Data/2020_periode_1/morrisons_ml_opinions.csv /tmp/morrisons_ml_opinions_topics.csv
!gsutil cp gs://di_data_sas/EN/UK/Ocado/Data/2020_periode_1/ocado_ml_opinions.csv  /tmp/ocado_ml_opinions_topics.csv
!gsutil cp gs://di_data_sas/FR/Coursesu/Data/2020_periode_10/coursesu_ml_opinions.csv /tmp/courseu.csv

edited_types = {
'asin':         'object',
'average'  :        'float16',
'review_body'     :  'object',
'review_date'      : 'object',
'review_likes'     : 'object',
'review_rating'    :'float16',
'review_title'     : 'object',
'five_star'        :'float16',
'four_star'        :'float16',
'one_star'        :'float16',
'pp_date'          : 'object',
'three_star'       :'float16',
'two_star'         :'float16',
'refpe'            : 'object',
'text_clean'        :'object',
'title_clean'       :'object',
'ml_score'         :'float16',
'text'             : 'object',
'ml_topic'         : 'category',
'opinion'          : 'object',
}

list_retailers = ['Amazon' , 'Asda', 'Morrisons', 'Ocado', 'Target', 'Walmart']
retailers = {}
for retailer in list_retailers:

    retailers[retailer] = pd.read_csv('/tmp/' + retailer.lower() + 
                                      '_ml_opinions_topics.csv', dtype=edited_types, nrows=10000)
    print(retailer)
    retailers[retailer] = retailers[retailer].dropna(subset=['review_body'])
    
to_concat = [retailers[retailer][['review_body', 'review_rating']] for retailer in list_retailers]
#to_concat = [retailers[retailer]['review_body'] for retailer in list_retailers]
data = pd.concat(to_concat, ignore_index = True)
"""


# getting data
class DataStructure(Dataset):

    def __init__(self, dataframe, ratio, tokenizer, loading_mode='Dataframe'):
        # loading mode
        self.loading_mode = loading_mode
        if self.loading_mode == 'file':
            self.df = pd.read_csv(dataframe)
        elif self.loading_mode == 'generator':
            # TO DO
            pass
        else:
            self.df = dataframe[['review_body', 'review_rating']]
        self.ratio = ratio
        self.tokenizer = tokenizer

    # generator to apply functions on dataframe
    def filter_df(self):
        def apply_on_df(*a, **kw):
            def apply(func):
                df = kw['df']
                column = kw['column']
                df[column] = df[column].apply(lambda x: func(x))
            return apply

        @apply_on_df(df=self.df, column='review_body')
        def remove_punctuation(x):
            for char in punctuation:
                x = str(x).replace(char, '')
            return x

    def generate_data(self):
        self.filter_df()
        # split dataframe into singles dataframes for each rating score
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        data_1 = self.df.loc[lambda df: df['review_rating'] == 1]
        data_2 = self.df.loc[lambda df: df['review_rating'] == 2]
        data_3 = self.df.loc[lambda df: df['review_rating'] == 3]
        data_4 = self.df.loc[lambda df: df['review_rating'] == 4]
        data_5 = self.df.loc[lambda df: df['review_rating'] == 5]

        # spliting each score dataframe into two dataframes set by a ratio
        data_val_1 = data_1[:int(self.ratio*len(data_1))]
        data_train_1 = data_1[int(self.ratio*len(data_1)):]

        data_val_2 = data_2[:int(self.ratio*len(data_2))]
        data_train_2 = data_2[int(self.ratio*len(data_2)):]

        data_val_3 = data_3[:int(self.ratio*len(data_3))]
        data_train_3 = data_3[int(self.ratio*len(data_3)):]

        data_val_4 = data_4[:int(self.ratio*len(data_4))]
        data_train_4 = data_4[int(self.ratio*len(data_4)):]

        data_val_5 = data_5[:int(self.ratio*len(data_5))]
        data_train_5 = data_5[int(self.ratio*len(data_5)):]

        # concat dfs split by ratio
        train_x = pd.concat([data_train_1, data_train_2,
                             data_train_3,  data_train_4, data_train_5])
        val_x = pd.concat(
            [data_val_1, data_val_2, data_train_3, data_val_4, data_val_5])

        # setting positifs 1 for rating >3
        train_x['score'] = train_x['review_rating'].apply(
            lambda x: 1 if x > 3 else 0)
        val_x['score'] = val_x['review_rating'].apply(
            lambda x: 1 if x > 3 else 0)

        df_train = train_x[['review_body', 'score']]
        df_val = val_x[['review_body', 'score']]

        # store the tokens length for every review
        token_lens = []
        for txt in self.df.review_body:
            _tokens = self.tokenizer.encode(txt, max_length=256)
            token_lens.append(len(_tokens))

        # shuffle the datasets
        df_train = df_train.sample(frac=1).reset_index(drop=True)
        df_val = df_val.sample(frac=1).reset_index(drop=True)
        return df_train, df_val, token_lens


class ReviewScoreDataset(Dataset):

    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        tokens = self.tokenizer.tokenize(review)
        bos_token = self.tokenizer.cls_token
        eos_token = self.tokenizer.sep_token
        tokens = [bos_token] + tokens + [eos_token]
        if len(tokens) < self.max_len:
            tokens = tokens + \
                [self.tokenizer.pad_token for _ in range(
                    self.max_len - len(tokens))]
        else:
            tokens = tokens[:self.max_len-1] + [eos_token]
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tokens_ids_tensor = torch.tensor(tokens_ids)
        attn_mask = torch.tensor((tokens_ids_tensor != 0).long())
        return {
            'review_text': review,
            'input_ids': tokens_ids_tensor.flatten(),
            'attention_mask': attn_mask.flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewScoreDataset(
        reviews=df.review_body.to_numpy(),
        targets=df.score.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=1
    )


train_data_loader = create_data_loader(
    df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)


# freeze all the parameters
for param in bert_model.parameters():
    param.requires_grad = False


class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(768, 512)  # 768 hidden state of bert
        self.drop2 = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)  # nclass = 2
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        x = self.dense1(pooled_output[1])
        x = self.relu(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.out(x)
        x = self.softmax(x)
        return x


model = SentimentClassifier(len(class_names))
model = model.to(device)


EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=True)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)


class train_model():
    def __init__(self, model, train_data_loader, len_train_data, val_data_loader, len_val_data, loss_fn, optimizer, device, scheduler, n_epochs=10):
        self.model = model
        self.train_data_loader = train_data_loader
        self.len_train_data = len_train_data
        self.val_data_loader = val_data_loader
        self.len_val_data = len_val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.epochs = n_epochs

    def train_epoch(self):
        model = self.model.train()

        losses = []
        correct_predictions = 0

        for d in self.train_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return correct_predictions.double() / self.len_train_data, np.mean(losses)

    @torch.no_grad()
    def eval_model(self):
        model = self.model.eval()

        losses = []
        correct_predictions = 0

        for d in self.val_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = self.loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

        return correct_predictions.double() / self.len_val_data, np.mean(losses)

    def run(self):
        _best_accuracy = 0
        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('↓→←↓'*20)
            train_acc, train_loss = self.train_epoch()
            print(f'Train loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = self.eval_model()
            print(f'Val   loss {val_loss} accuracy {val_acc}')

            if val_acc > _best_accuracy:
                torch.save(model.state_dict(), 'score_model.bin')
                _best_accuracy = val_acc


"""run training 

%%time
torch.cuda.empty_cache()
sentiment_object = train_model(model, train_data_loader, len(df_train), val_data_loader, len(df_val), loss_fn, optimizer, device, scheduler, EPOCHS)
sentiment_object.run()
"""

# load weights of best model
"""
path = '/content/drive/MyDrive/score_model.bin'
model.load_state_dict(torch.load(path))"""

# evalutation


def eval_model(model, data_loader, loss_fn, device, n_samples):
    model = model.eval()

    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_samples, np.mean(losses)


# prediction
@torch.no_grad()
def get_predictions(model, data_loader):
    model = model.eval()

    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    # with torch.no_grad():
    for d in data_loader:

        texts = d["review_text"]
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)

        probs = F.softmax(outputs, dim=1)

        review_texts.extend(texts)
        predictions.extend(preds)
        prediction_probs.extend(probs)
        real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values


# prediction on one review (raw text)
def live_prediction(tokenizer, sentence, max_len=MAX_LEN):
    encoded_review = tokenizer.encode_plus(
        sentence,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    class_names = ['Negative', 'Positive']
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    # print(output)
    _, prediction = torch.max(output, dim=1)
    print(_, prediction)
    results = {}
    results['Review_text'] = sentence
    results['Sentiment'] = class_names[prediction]
    return results


# using fintuned transformers model
class NlptownPrediction():

    def __init__(self, dataframe):
        self.df = dataframe

    # generator to apply functions on dataframe

    def filter_df(self):
        def apply_on_df(*a, **kw):
            def apply(func):
                df = kw['df']
                column = kw['column']
                df[column] = df[column].apply(lambda x: func(x))
            return apply

        @apply_on_df(df=self.df, column='review')
        def remove_punctuation(x):
            for char in punctuation:
                x = str(x).replace(char, '')
            return x

        classifier = EasySequenceClassifier()

    @staticmethod
    def predict_sentiment_rating(sentence):
        sentences = classifier.tag_text(
            text=sentence,
            model_name_or_path="nlptown/bert-base-multilingual-uncased-sentiment",
            mini_batch_size=1,
        )
        labels_list = [item.to_dict() for item in sentences[0].labels]
        new_dict = {
            item['value'].split(' ')[0]: item['confidence'] for item in labels_list}
        predicted_rating = max(new_dict, key=new_dict.get)
        predicted_score = 1 if int(predicted_rating) > 3 else 0
        return predicted_rating, predicted_score

    def generate_data(self):
        # self.filter_df()

        # setting positifs 1 for rating >3
        df['score'] = df['review_rating'].apply(lambda x: 1 if x > 3 else 0)
        df['predicted_score'] = df['review'].apply(
            lambda x: self.predict_sentiment_rating(x)[1])
        df['predicted_rating'] = df['review'].apply(
            lambda x: self.predict_sentiment_rating(x)[0])
        return df

if __name__ == "__main__":
    pass