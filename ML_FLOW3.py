import pandas as pd
import numpy as np
import language_tool_python
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.decomposition import PCA

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk
nltk.download('wordnet')

import stanza
import re
from nltk.corpus import stopwords
import es_core_news_sm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

import stanza
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

#"" Número de stopwords en nltk
stop_nltk = stopwords.words('spanish')
print("nltk :",len(stop_nltk))

## Número de stopwords en spacy
nlp = es_core_news_sm.load()
stop_spacy = nlp.Defaults.stop_words
print("spacy:", len(stop_spacy))
stop_todas = list(stop_spacy.union(set(stop_nltk)))

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import spacy

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional, GRU, Input,Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import plot_model

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBClassifier
from scipy import stats
from itertools import combinations

class ML_FLOW_PARCIAL3:
    def __init__(self):
        self.path = "/content/drive/MyDrive/QUIZ1_3CORTE_IA/TercerPunto/"
        
    def load_data(self):
        # Utiliza el atributo self.path
        
        df = pd.read_csv("Sarcasmo_train.csv", sep=";", encoding="UTF-8")
        
        # Mapeo de categorías
        def categoria_y(x):
            return 0 if x == "Si" else 1

        df["Sarcasmo"] = df["Sarcasmo"].map(categoria_y)
        
        # Cargar y procesar el archivo de prueba
        pru = pd.read_csv("Sarcasmo_test.csv", sep=";", encoding="UTF-8")
        
        pru["Sarcasmo"] = pru["Sarcasmo"].map(categoria_y)
        
        return df, pru
    
    def preprocessing(self):
        #tool = language_tool_python.LanguageTool('es')

        ## Datos sin ingenieria de variables
        datos = pd.read_pickle("datos.pkl")
        datos['processed_text'] = datos.apply(lambda row:  ' '.join(token.lemma_ for token in nlp(row["locucion_c"]).sents), axis=1)
        datos['processed_text'] = datos['processed_text'].str.lower()
        datos['processed_text'] = datos['processed_text'].replace(list('áéíóú'),list('aeiou'),regex=True)
        datos['processed_text'] = datos['processed_text'].str.replace('[^\w\s]','')
        datos['processed_text'] = datos['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_todas)]))
        #self.datos = datos
        
        datos_pru = pd.read_pickle("datos_pru.pkl")
        datos_pru['processed_text'] = datos_pru.apply(lambda row:  ' '.join(token.lemma_ for token in nlp(row["locucion_c"]).sents), axis=1)
        datos_pru['processed_text'] = datos_pru['processed_text'].str.lower()
        datos_pru['processed_text'] = datos_pru['processed_text'].replace(list('áéíóú'),list('aeiou'),regex=True)
        datos_pru['processed_text'] = datos_pru['processed_text'].str.replace('[^\w\s]','')
        datos_pru['processed_text'] = datos_pru['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_todas)]))
        #self.datos_pru = datos_pru
        
        # Datos con ingenieria de variables
        base_modelo = pd.read_pickle("base_modelo.pkl")
        base_modelo['processed_text'] = base_modelo.apply(lambda row:  ' '.join(token.lemma_ for token in nlp(row["locucion_c"]).sents), axis=1)
        base_modelo['processed_text'] = base_modelo['processed_text'].str.lower()
        base_modelo['processed_text'] = base_modelo['processed_text'].replace(list('áéíóú'),list('aeiou'),regex=True)
        base_modelo['processed_text'] = base_modelo['processed_text'].str.replace('[^\w\s]','')
        base_modelo['processed_text'] = base_modelo['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_todas)]))
        #self.base_modelo = base_modelo
        
        base_modelo_p = pd.read_pickle("base_modelo_p.pkl")
        base_modelo_p['processed_text'] = base_modelo_p.apply(lambda row:  ' '.join(token.lemma_ for token in nlp(row["locucion_c"]).sents), axis=1)
        base_modelo_p['processed_text'] = base_modelo_p['processed_text'].str.lower()
        base_modelo_p['processed_text'] = base_modelo_p['processed_text'].replace(list('áéíóú'),list('aeiou'),regex=True)
        base_modelo_p['processed_text'] = base_modelo_p['processed_text'].str.replace('[^\w\s]','')
        base_modelo_p['processed_text'] = base_modelo_p['processed_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_todas)]))
        #self.base_modelo_p = base_modelo_p
        print

        return datos, datos_pru, base_modelo, base_modelo_p
        
    def train_model(self):
        ## MODELO BERT
        
        # Cargar el modelo BERT preentrenado para español
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        
        ## Embeddings sin ingenieria
        doc_embedding = pd.read_pickle("embeddings.pkl")
        doc_embedding_pru = pd.read_pickle("embeddings_pru.pkl")

        ## Embeddings con ingenieria
        doc_embedding_ing = pd.read_pickle("embeddings_ing.pkl")
        doc_embedding_pru_ing = pd.read_pickle("embeddings_pru_ing.pkl")

        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Usar la representación de la [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return cls_embedding.flatten()

        # Aplicar BERT a cada texto
        self.datos['bert_embedding'] = self.datos['processed_text'].apply(get_bert_embedding)

        # Obtener embeddings con BERT
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Usar la representación de la [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return cls_embedding.flatten()

        # Aplicar BERT a cada texto
        self.datos_pru['bert_embedding'] = self.datos_pru['processed_text'].apply(get_bert_embedding)      

        ## Modelo supervisado (Red neuronal densa FCNN) sin ingenieria
        X_train = self.datos.drop(columns=["Sarcasmo", "locucion_c", "tokens", "processed_text", "bert_embedding"])
        y_train = self.datos["Sarcasmo"]
        X_test = self.datos_pru.drop(columns=["Sarcasmo", "locucion_c", "tokens", "processed_text", "bert_embedding"])
        y_test = self.datos_pru["Sarcasmo"]

        scaler = MinMaxScaler((-1.0,1.0))
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
        X_train_scaled.columns = X_train.columns
        X_test_scaled = pd.DataFrame(scaler.transform(X_test))
        X_test_scaled.columns = X_test.columns

        embedding_vector_length = 768

        x1 = Input(shape=(embedding_vector_length,), name='Input_Embedding')
        x2 = Input(shape=(X_train_scaled.shape[1],), name='Input_Features')

        # Capa entrada
        x = Concatenate(name='Concatenar')([x1, x2])
        x = Dropout(0.25)(x)

        # capas ocultas
        x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
        x = Dropout(0.25)(x)

        # Capa de salida para clasificación binaria
        x = Dense(1, activation='sigmoid', name='Output')(x)

        model = Model(inputs=[x1, x2], outputs=x)

        # Compilación para clasificación binaria
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        embeddings_train = doc_embedding
        embeddings_test = doc_embedding_pru

        history = model.fit(x = [embeddings_train,X_train_scaled],
                            y = y_train,
                            validation_data = ([embeddings_test,X_test_scaled],y_test),
                            epochs=100,
                            batch_size=32,verbose=1)

        y_pred = model.predict([embeddings_test,X_test_scaled])
        y_true = y_test
        y_pred = (y_pred>=0.5).astype(int)
        accuracy_1 = accuracy_score(y_true,y_pred)
        roc_auc_1 = roc_auc_score(y_true,y_pred)


        ## Modelo Bert con ingenieria
        # Obtener embeddings con BERT
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')

        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Usar la representación de la [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return cls_embedding.flatten()

        # Aplicar BERT a cada texto
        self.base_modelo['bert_embedding'] = self.base_modelo['processed_text'].apply(get_bert_embedding)

        # Obtener embeddings con BERT
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            # Usar la representación de la [CLS] token
            cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
            return cls_embedding.flatten()

        # Aplicar BERT a cada texto
        self.base_modelo_p['bert_embedding'] = self.base_modelo_p['processed_text'].apply(get_bert_embedding)

        ## Modelo supervisado (Red neuronal densa FCNN) con ingenieria
        X_train = self.base_modelo.drop(columns=["Sarcasmo", "locucion_c", "tokens", "processed_text", "bert_embedding"])
        y_train = self.base_modelo["Sarcasmo"]
        X_test = self.base_modelo_p.drop(columns=["Sarcasmo", "locucion_c", "tokens", "processed_text", "bert_embedding"])
        y_test = self.base_modelo_p["Sarcasmo"]

        scaler = MinMaxScaler((-1.0,1.0))
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
        X_train_scaled.columns = X_train.columns
        X_test_scaled = pd.DataFrame(scaler.transform(X_test))
        X_test_scaled.columns = X_test.columns

        embedding_vector_length = 768

        x1 = Input(shape=(embedding_vector_length,), name='Input_Embedding')
        x2 = Input(shape=(X_train_scaled.shape[1],), name='Input_Features')

        # Capa entrada
        x = Concatenate(name='Concatenar')([x1, x2])
        x = Dropout(0.25)(x)

        # capas ocultas
        x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
        x = Dropout(0.25)(x)

        # Capa de salida para clasificación binaria
        x = Dense(1, activation='sigmoid', name='Output')(x)

        model = Model(inputs=[x1, x2], outputs=x)

        # Compilación para clasificación binaria
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        embeddings_train = doc_embedding_ing
        embeddings_test = doc_embedding_pru_ing

        history = model.fit(x = [embeddings_train,X_train_scaled],
                            y = y_train,
                            validation_data = ([embeddings_test,X_test_scaled],y_test),
                            epochs=100,
                            batch_size=32,verbose=1)

        y_pred = model.predict([embeddings_test,X_test_scaled])
        y_true = y_test
        y_pred = (y_pred>=0.5).astype(int)
        accuracy_2 = accuracy_score(y_true,y_pred)
        roc_auc_2 = roc_auc_score(y_true,y_pred)

        ## MODELO FAST TEXT
        data = pd.read_pickle("datos.pkl")
        data_pru = pd.read_pickle("datos_pru.pkl")

        data_ing = pd.read_pickle("base_modelo.pkl")
        data_pru_ing = pd.read_pickle("base_modelo_p.pkl")

        ## Embeddings sin ingenieria
        doc_embedding_f = pd.read_pickle("embeddings_f.pkl")
        doc_embedding_pru_f = pd.read_pickle("embeddings_pru_f.pkl")

        ## Embeddings con ingenieria
        doc_embedding_fing = pd.read_pickle("embedding_fing.pkl")
        doc_embedding_pru_fing = pd.read_pickle("embedding_pru_fing.pkl")

        ## Modelo supervisado (Red neuronal densa FCNN) sin ingenieria
        X_train = data.drop(columns=["Sarcasmo", "locucion_c", "tokens"])
        y_train = data["Sarcasmo"]
        X_test = data_pru.drop(columns=["Sarcasmo", "locucion_c", "tokens"])
        y_test = data_pru["Sarcasmo"]

        scaler = MinMaxScaler((-1.0,1.0))
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
        X_train_scaled.columns = X_train.columns
        X_test_scaled = pd.DataFrame(scaler.transform(X_test))
        X_test_scaled.columns = X_test.columns

        embedding_vector_length = 300

        x1 = Input(shape=(embedding_vector_length,), name='Input_Embedding')
        x2 = Input(shape=(X_train_scaled.shape[1],), name='Input_Features')

        # Capa entrada
        x = Concatenate(name='Concatenar')([x1, x2])
        x = Dropout(0.25)(x)

        # capas ocultas
        x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
        x = Dropout(0.25)(x)

        # Capa de salida para clasificación binaria
        x = Dense(1, activation='sigmoid', name='Output')(x)

        model = Model(inputs=[x1, x2], outputs=x)

        # Compilación para clasificación binaria
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        embeddings_train = doc_embedding_f
        embeddings_test = doc_embedding_pru_f

        history = model.fit(x = [embeddings_train,X_train_scaled],
                            y = y_train,
                            validation_data = ([embeddings_test,X_test_scaled],y_test),
                            epochs=100,
                            batch_size=32,verbose=1)

        y_pred = model.predict([embeddings_test,X_test_scaled])
        y_true = y_test
        y_pred = (y_pred>=0.5).astype(int)
        accuracy_3 = accuracy_score(y_true,y_pred)
        roc_auc_3 = roc_auc_score(y_true,y_pred)

        ## Modelo supervisado (Red neuronal densa FCNN) con ingenieria
        X_train = data_ing.drop(columns=["Sarcasmo", "locucion_c", "tokens"])
        y_train = data_ing["Sarcasmo"]
        X_test = data_pru_ing.drop(columns=["Sarcasmo", "locucion_c", "tokens"])
        y_test = data_pru_ing["Sarcasmo"]

        scaler = MinMaxScaler((-1.0,1.0))

        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
        X_train_scaled.columns = X_train.columns
        X_test_scaled = pd.DataFrame(scaler.transform(X_test))
        X_test_scaled.columns = X_test.columns

        embedding_vector_length = 300

        x1 = Input(shape=(embedding_vector_length,), name='Input_Embedding')
        x2 = Input(shape=(X_train_scaled.shape[1],), name='Input_Features')

        # Capa entrada
        x = Concatenate(name='Concatenar')([x1, x2])
        x = Dropout(0.25)(x)

        # capas ocultas
        x = Dense(64, activation='elu', name='Capa_Densa_1')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='elu', name='Capa_Densa_2')(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='elu', name='Capa_Densa_3')(x)
        x = Dropout(0.25)(x)

        # Capa de salida para clasificación binaria
        x = Dense(1, activation='sigmoid', name='Output')(x)

        model = Model(inputs=[x1, x2], outputs=x)

        # Compilación para clasificación binaria
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()

        embeddings_train = doc_embedding_fing
        embeddings_test = doc_embedding_pru_fing

        history = model.fit(x = [embeddings_train,X_train_scaled],
                            y = y_train,
                            validation_data = ([embeddings_test,X_test_scaled],y_test),
                            epochs=100,
                            batch_size=32,verbose=1)

        y_pred = model.predict([embeddings_test,X_test_scaled])
        y_true = y_test
        y_pred = (y_pred>=0.5).astype(int)
        accuracy_4 = accuracy_score(y_true,y_pred)
        roc_auc_4 = roc_auc_score(y_true,y_pred)

        return accuracy_4, roc_auc_4, accuracy_3, roc_auc_3, accuracy_2, roc_auc_2, accuracy_1, roc_auc_1
        
    def predict(self, accuracy_4, roc_auc_4, accuracy_3, roc_auc_3, accuracy_2, roc_auc_2, accuracy_1, roc_auc_1):
        
        modelos = ['BERT_Sin_Eng','BERT_Con_Eng','FastText_Sin_Eng','FastText_Con_Eng']
        accuracy_scores = [accuracy_1, accuracy_2, accuracy_3, accuracy_4]
        roc_auc_scores = [roc_auc_1, roc_auc_2, roc_auc_3, roc_auc_4]

        # Crear el DataFrame con los resultados
        resultados = pd.DataFrame({
            'Modelo': modelos,
            'Accuracy': accuracy_scores,
            'ROC_AUC': roc_auc_scores
        })

        resultados.to_csv(self.path + 'TABLA METRICAS.csv')

        mejor_modelo = resultados.sort_values(by='ROC_AUC', ascending=False).iloc[0]

        return mejor_modelo
        
    def ML_FLOW(self):
        try:
            # Paso 1: Cargar datos
            df, pru = self.load_data()
        
            # Paso 2: Preprocesamiento
            self.datos, self.datos_pru, self.base_modelo, self.base_modelo_p = self.preprocessing()
        
            # Paso 3: Entrenamiento del modelo
            accuracy_4, roc_auc_4, accuracy_3, roc_auc_3, accuracy_2, roc_auc_2, accuracy_1, roc_auc_1 = self.train_model()
         
            # Paso 4: Predicción
            mejor_modelo = self.predict(accuracy_4, roc_auc_4, accuracy_3, roc_auc_3, accuracy_2, roc_auc_2, accuracy_1, roc_auc_1)

            mensaje = f"El modelo con el mayor ROC AUC es: {mejor_modelo['Modelo']} con un ROC AUC de {mejor_modelo['ROC_AUC']}"

            return {'success':True,'message':mensaje}
        except Exception as e:
            return {'success':False,'message':str(e)}