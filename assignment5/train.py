import math
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the digits dataset
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# y = digits.target
# 
# # Create the RFE object and rank each pixel
# svc = SVC(kernel="linear", C=1)
# rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
# rfe.fit(X, y)
# ranking = rfe.ranking_.reshape(digits.images[0].shape)
# 
# # Plot pixel ranking
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()


# Globals
BASELINE = False


def standardize(data, target, s=None):
    if not s:
        s = [preprocessing.StandardScaler(),
             preprocessing.StandardScaler()]
    s1, s2 = s
    data = s1.fit_transform(data) 
    # target = s1.fit_transform(target) 

    return data, target, [s1, s2]


def prepare_data(path, key, s=None):
    df = shuffle(pd.read_table(path))
    # target = df[key].values.reshape(-1, 1)
    target = df[key].values
    data_df = df.drop(["student_ID", "english_grade", "math_grade", "german_grade", "lang_iq", "logic_iq", "image_no", "UUID", "MIX_text"] , 1)
    return standardize(data_df.values, target, s), list(data_df.columns)


def prepare_baseline_data(path, key, s=None):
    df = shuffle(pd.read_table(path))
    # target = df[key].values.reshape(-1, 1)
    target = df[key].values
    # data = df[["Surface Measures - nParagraphs", "Surface Measures - nSentences", "Surface Measures - nTokens"]].values
    data_df = df[["Surface Measures - nTokens"]]
    return standardize(data_df.values, target, s), list(data_df.columns)


def train(key):
    
    if BASELINE:
        (trainX, trainY, s), columns = prepare_baseline_data("./training_data-" + key + ".tsv", key)
        (validX, validY, _), columns = prepare_baseline_data("./development_data-" + key + ".tsv", key, s)
        (testX, testY, _), columns = prepare_baseline_data("./test_data-" + key + ".tsv", key, s)
    else:
        (trainX, trainY, s), columns = prepare_data("./training_data-" + key + ".tsv", key)
        (validX, validY, _), columns = prepare_data("./development_data-" + key + ".tsv", key, s)
        (testX, testY, _), columns = prepare_data("./test_data-" + key + ".tsv", key, s)
    # (X, y, _), columns = prepare_baseline_data("./training_data-" + key + ".tsv", key)

    # Create the RFE object and rank each pixel
    # reg = linear_model.LinearRegression()
    reg = SVR()
    # svc = SVC(kernel="linear", C=1)
    rfecv = RFE(estimator=reg, n_features_to_select=1, step=1)
    rfecv.fit(trainX, trainY)

    # rfecv = RFECV(estimator=reg, step=1,
    #           scoring="neg_mean_squared_error")
    # rfecv.fit(trainX, trainY)

    print("Train Score: ", rfecv.score(trainX, trainY))
    print("Validation Score: ", rfecv.score(validX, validY))
    print("Test Score: ", rfecv.score(testX, testY))

    # print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()


    # return rfecv, columns
    return rfecv, columns
    
    