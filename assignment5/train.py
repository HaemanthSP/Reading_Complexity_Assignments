import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing
from tensorflow.keras import layers
import math


# Globals
BASELINE = True


def create_NN_model():
    model = tf.keras.Sequential([
      layers.Dense(50, activation='relu'),
      layers.Dense(10, activation='relu'),
      layers.Dense(1)])
    return model

def standardize(data, target):
    s1 = preprocessing.MinMaxScaler(feature_range=(0,10))
    # scaler = preprocessing.StandardScaler()
    # data = s1.fit_transform(data) 
    # data = scaler.fit_transform(data)
    s2 = preprocessing.MinMaxScaler(feature_range=(0,4))
    # target = s2.fit_transform(target)
    return data, target, s1, s2 

def create_svm_model():
    # model = svm.SVC(kernel="rbf")
    model = svm.SVR()
    return model

def prepare_data(path):
    df = shuffle(pd.read_table(path))
    target = df.pop("lang_iq").values.reshape(-1, 1)
    data = df.drop(["student_ID", "english_grade", "math_grade", "german_grade", "logic_iq", "image_no", "UUID", "MIX_text"] , 1).values
    # data = df.drop(["student_ID", "english_grade", "math_grade", "lang_iq", "logic_iq", "image_no", "UUID", "MIX_text"] , 1).values
    return standardize(data, target)


def prepare_baseline_data(path):
    df = shuffle(pd.read_table(path))
    target = df.pop("lang_iq").values.reshape(-1, 1)
    data = df[["Surface Measures - nParagraphs", "Surface Measures - nSentences", "Surface Measures - nTokens"]].values
    return standardize(data, target)

    
def train():
    if BASELINE:
        trainX, trainY, s1, s2 = prepare_baseline_data("./training_data-lang_iq.tsv")
        validX, validY, _, _ = prepare_baseline_data("./development_data-lang_iq.tsv")
        testX, testY, _, _ = prepare_baseline_data("./test_data-lang_iq.tsv")
    else:
        trainX, trainY, s1, s2 = prepare_data("./training_data-lang_iq.tsv")
        validX, validY, _, _ = prepare_data("./development_data-lang_iq.tsv")
        testX, testY, _, _ = prepare_data("./test_data-lang_iq.tsv")
        
    model = create_NN_model()
    model.compile(optimizer='adam',
              loss='mse')
              # metrics=[])
    model.fit(trainX, trainY, epochs=50, validation_data=[validX, validY])
    
    eval_loss = model.evaluate(testX, testY)
    print('\nEval loss: {:.3f}'.format(eval_loss))
    return model