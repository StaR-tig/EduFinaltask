import flask
import os
import pickle
import re
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
import torch
from transformers import AutoTokenizer, AutoModel

#функция обработки текста
def preprocessing_text(text):
    text = text.lower().replace('ё', 'е')
    text = re.sub('((www\.[^\s]+)\|(https?://[^\s]+))', 'URL',text)
    text = re.sub('[^a-zA-Za-яА-Я]+',' ', text)
    text = re.sub(' +',' ', text)
    
    return text.strip()

#подгрузка модели для формирования эмбеддингов
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny")

def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    
    return embeddings[0].cpu().numpy()


#загрузка классификациорнной модели
save_path = r"model\KNN_9_model.pkl"
loaded_knn_model = pickle.load(open(save_path, 'rb'))

#обработка комментария
def tone_recognition(comment):
    emb = embed_bert_cls(preprocessing_text(comment),model,tokenizer)
    if loaded_knn_model.predict([np.stack(emb, axis=0 )]) == 0:
        return "Негативный комментарий"
    else:
        return "Положительный комментарий"


