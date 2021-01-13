# Présentation

Sentiment Analysis on e-commerce Reviews using Deep L algorithms (BERT, xnli Roberta, LSTM)

# Installation de librairies requies

pip install -r requirements.txt

# Lancement du server

### run elasticsearch server
! cd bdd/ealsticsearch
! ./bin/elasticsearch
### run server
python main.py

### run both
./../bdd/elasticsearch-7.10.0/bin/elasticsearch & python main.py
### clear the ports
fuser -n tcp -k 4000
fuser -n tcp -k 9200

# Acces au service
http://172.28.100.53:4000

# ou en local
http://localhost:4000

