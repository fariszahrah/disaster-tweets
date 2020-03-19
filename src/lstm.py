
from clean_tweet import clean

from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Conv2D, MaxPooling2D, Bidirectional,  Reshape, Flatten
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import load_model

import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

import os

import pandas as pd
import re
from wordcloud import STOPWORDS

#All Text Cleaning Credit: Gunes Evitan
#Ty sir, this is alot of hours you have saved me and others


train = pd.read_csv('./../data/train.csv')
test = pd.read_csv('./../data/test.csv')


#shuffle and reset index
state = 24
train_df = train.sample(frac=1,random_state=state)
test_df = test.sample(frac=1,random_state=state)
train_df.reset_index(inplace=True, drop=True)
test_df.reset_index(inplace=True, drop=True) 

#cleaning
def preprocessor2(text):
    text = text.replace('%20',' ')
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = (re.sub('[^a-zA-Z0-9_]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


#more cleaning
def remove_url(raw_str):
    clean_str = re.sub(r'http\S+', '', raw_str)
    return clean_str

#more cleaning
def preprocessor3(text):
    text = re.sub(r'^washington d c ', "washington dc", text)
    text = re.sub(r'^washington +[\w]*', "washington dc", text)
    text = re.sub(r'^new york +[\w]*', "new york", text)
    text = re.sub(r'^nyc$', "new york", text)
    text = re.sub(r'^chicago +[\w]*', "chicago", text)
    text = re.sub(r'^california +[\w]*', "california", text)
    text = re.sub(r'^los angeles +[\w]*', "los angeles", text)
    text = re.sub(r'^san francisco +[\w]*', "san francisco", text)
    text = re.sub(r'^london +[\w]*', "london", text)
    text = re.sub(r'^usa$', "united states", text)
    text = re.sub(r'^us$', "united states", text)
    text = re.sub(r'^uk$', "united kingdom", text)

    return text


#more cleaning
def preprocessor4(text):
    abb = ['ak', 'al', 'az', 'ar', 'ca', 'co',
           'ct', 'de', 'dc', 'fl', 'ga', 'hi',
           'id', 'il', 'in', 'ia', 'ks', 'ky',
           'la', 'me', 'mt', 'ne', 'nv', 'nh',
           'nj', 'nm', 'ny', 'nc', 'nd', 'oh',
           'ok', 'or', 'md', 'ma', 'mi', 'mn',
           'ms', 'mo', 'pa', 'ri', 'sc', 'sd',
           'tn', 'tx', 'ut', 'vt', 'va', 'wa',
           'wv', 'wi', 'wy']

    for i in abb:
        text = re.sub(r' {0}$'.format(i), '', text)

        return text


train_df['text'] = train_df['text'].apply(lambda x:remove_url(x))
train_df['keyword_cleaned'] = train_df['keyword'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x))
train_df['location_cleaned'] = train_df['location'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x))
train_df['text_cleaned'] = train_df['text'].copy().apply(lambda x : clean(x)).apply(lambda x : preprocessor2(x))

test_df['text'] = test_df['text'].apply(lambda x:remove_url(x))
test_df['keyword_cleaned'] = test_df['keyword'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x))
test_df['location_cleaned'] = test_df['location'].copy().apply(lambda x : clean(str(x))).apply(lambda x : preprocessor2(x))
test_df['text_cleaned'] = test_df['text'].copy().apply(lambda x : clean(x)).apply(lambda x : preprocessor2(x))

print(f'train_df columns: {train_df.columns}')

#actually callign as the cleaning functions 
train_df['location_cleaned'] = train_df['location_cleaned'].copy().apply(
        lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
train_df['text_cleaned'] = train_df['text_cleaned'].copy().apply(
        lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
test_df['location_cleaned'] = test_df['location_cleaned'].copy().apply(
        lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))
test_df['text_cleaned'] = test_df['text_cleaned'].copy().apply(
        lambda x : preprocessor3(x)).apply(lambda x : preprocessor4(x))



#stop words, testing if these help or not
stopwords = list(STOPWORDS)+['will','may','one','now','nan','don']

train_df['text_cleaned'] = train_df['text_cleaned'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
test_df['text_cleaned'] = test_df['text_cleaned'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))



temp_df1 = list(zip(train_df.target,train_df.keyword_cleaned,
                    train_df.location_cleaned,train_df.text_cleaned))
temp_df2 = list(zip(train_df.target,train_df.keyword_cleaned,
                    train_df.location_cleaned,train_df.text_cleaned))
x = pd.DataFrame(temp_df1, columns =
        ['target','keyword_cleaned','location_cleaned','text_cleaned'])
y = pd.DataFrame(temp_df2, columns =
        ['target','keyword_cleaned','location_cleaned','text_cleaned'])


z = pd.concat([train_df,x,y], axis=0, join='outer', ignore_index=False, keys=None, sort = False)
z = z[['id','keyword_cleaned','location_cleaned','text_cleaned','target']].copy()
z.reset_index(inplace=True, drop=True)



train_df = z 
test_df = test_df[['id','keyword_cleaned','location_cleaned','text_cleaned']].copy()


#Randomization
state = 1
train_df = train_df.sample(frac=1,random_state=state)
train_df.reset_index(inplace=True, drop=True)

#tokenizing, and eventually embedding
top_word = 35000
tok = Tokenizer(num_words=top_word)
tok.fit_on_texts((train_df['text_cleaned']+train_df['keyword_cleaned']+train_df['location_cleaned']))

text_lengths = [len(x.split()) for x in (train_df.text_cleaned)]
max_words = max(text_lengths) + 1
max_words_ky = max([len(x.split()) for x in (train_df.keyword_cleaned)]) + 1
max_words_lc = max([len(x.split()) for x in (train_df.location_cleaned)]) + 1



#Training set
val_value = 5000

X_train_tx = tok.texts_to_sequences(train_df['text_cleaned'])
X_train_ky = tok.texts_to_sequences(train_df['keyword_cleaned'])
X_train_lc = tok.texts_to_sequences(train_df['location_cleaned'])

X_test_tx = tok.texts_to_sequences(test_df['text_cleaned'])
X_test_ky = tok.texts_to_sequences(test_df['keyword_cleaned'])
X_test_lc = tok.texts_to_sequences(test_df['location_cleaned'])


#categorical for keras
Y_train = train_df['target']
Y_train = to_categorical(Y_train)



X_train_tx = sequence.pad_sequences(X_train_tx, maxlen=max_words)
X_train_ky = sequence.pad_sequences(X_train_ky, maxlen=max_words_ky)
X_train_lc = sequence.pad_sequences(X_train_lc, maxlen=max_words_lc)

X_test_tx = sequence.pad_sequences(X_test_tx, maxlen=max_words)
X_test_ky = sequence.pad_sequences(X_test_ky, maxlen=max_words_ky)
X_test_lc = sequence.pad_sequences(X_test_lc, maxlen=max_words_lc)



embeddings_index = dict()
f = open(os.path.join('/Users/fariszahrah/Data/glove.twitter.27B/', 'glove.twitter.27B.200d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
embedding_matrix = zeros((top_word, embedding_dim))
for word, index in tok.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector



input1 = Input(shape=(max_words,))
embedding_layer1 = Embedding(top_word, 200, weights=[embedding_matrix], input_length=max_words, trainable=False)(input1)
dropout1 = Dropout(0.2)(embedding_layer1)
lstm1_1 = LSTM(128,return_sequences = True)(dropout1)
lstm1_2 = LSTM(128,return_sequences = True)(lstm1_1)
lstm1_2a = LSTM(128,return_sequences = True)(lstm1_2)
lstm1_3 = LSTM(128)(lstm1_2a)
res = Reshape((-1, X_train_tx.shape[1], 100))(lstm1_3)
conv1 = Conv2D(100, (3,3), padding='same',activation="relu")(res)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
flat1 = Flatten()(pool1)

input2 = Input(shape=(max_words_ky,))
embedding_layer2 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_ky, trainable=False)(input2)
lstm2_1 = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(embedding_layer2)
lstm2_1a = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm2_1)
lstm2_1b = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm2_1a)
res2 = Reshape((-1, X_train_ky.shape[1], 100))(lstm2_1b)
conv2 = Conv2D(100, (3,3), padding='same',activation="relu")(res2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
flat2 = Flatten()(pool2)

input3 = Input(shape=(max_words_lc,))
embedding_layer3 = Embedding(top_word, 100, weights=[embedding_matrix], input_length=max_words_lc, trainable=False)(input3)
lstm3_1 = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(embedding_layer3)
lstm3_1a = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm3_1)
lstm3_1b = Bidirectional(LSTM(100, return_sequences=True,dropout = 0.2))(lstm3_1a)
res3 = Reshape((-1, X_train_lc.shape[1], 100))(lstm3_1b)
conv3 = Conv2D(100, (3,3), padding='same',activation="relu")(res3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
flat3 = Flatten()(pool3)



merge = concatenate([flat1, flat2, flat3])


dropout = Dropout(0.4)(merge)
dense1 = Dense(256, activation='relu')(dropout)
dense2 = Dense(128, activation='relu')(dense1)
output = Dense(2, activation='softmax')(dense2)
model = Model(inputs=[input1,input2,input3], outputs=output)
model.summary()



model.compile(loss="binary_crossentropy", optimizer="adadelta",
              metrics=["accuracy"])

es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience = 4)
history = model.fit([X_train_tx,X_train_ky,X_train_lc], Y_train, validation_split=0.2, epochs=30, batch_size=64, verbose=2, callbacks=[es])




def result_eva (loss,val_loss,acc,val_acc):
    import matplotlib.pyplot as plt
    
    epochs = range(1,len(loss)+1)
    plt.plot(epochs, loss,'b-o', label ='Training Loss')
    plt.plot(epochs, val_loss,'r-o', label ='Validation Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, "b-o", label="Training Acc")
    plt.plot(epochs, val_acc, "r-o", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()





Y_pred = model.predict([X_test_tx], batch_size=64, verbose=2)
Y_pred = np.argmax(Y_pred,axis=1)

pred_df = pd.DataFrame(Y_pred, columns=['target'])
result = pd.concat([test_df,pred_df], axis=1, join='outer', ignore_index=False, keys=None, sort = False)
result = result[['id','target']]

result.to_csv('../submissions/lstm_text_kw_loc_submission.csv',index=False)
