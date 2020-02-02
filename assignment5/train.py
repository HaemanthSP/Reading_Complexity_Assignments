import math
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, make_scorer
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
BASELINE = True
SELECT_FEAT = False

def pearson_coeff(x, y):
    return pearsonr(x, y)[0]

def standardize(data, target, s=None):
    if not s:
        s = [preprocessing.StandardScaler(),
             preprocessing.StandardScaler()]
    s1, s2 = s
    data = s1.fit_transform(data) 
    # target = s1.fit_transform(target) 

    return data, target, [s1, s2]


def prepare_data(path, key, s=None):
    df = shuffle(pd.read_table(path), random_state=111)
    target = df[key].values
    if SELECT_FEAT:
        data_df = df.drop(["student_ID", "english_grade", "math_grade", "german_grade", "lang_iq", "logic_iq", "image_no", "UUID", "MIX_text"] , 1)
    else:
        with open('selected_features_' + key + '.txt', 'r') as f: 
            feats = [x.strip() for x in f.readlines()] 
        data_df = df[feats]
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


    reg = linear_model.Ridge()
    if SELECT_FEAT:
        # Create the RFE object and rank each pixel
        reg = linear_model.Ridge()
        # reg = SVR(kernel="linear")
        # svc = SVC(kernel="linear", C=1)
        # rfecv = RFE(estimator=reg, n_features_to_select=1, step=1)
        # rfecv.fit(trainX, trainY)

        # rfecv = RFECV(estimator=reg, step=1, scoring=make_scorer(pearson_coeff))
        rfecv = RFECV(estimator=reg, step=1, scoring="neg_mean_squared_error")
        rfecv.fit(trainX, trainY)

        print("Train Score: ", rfecv.score(trainX, trainY))
        print("Validation Score: ", rfecv.score(validX, validY))
        print("Test Score: ", rfecv.score(testX, testY))
        print("Optimal number of features : %d" % rfecv.n_features_)

        # Plot number of features VS. cross-validation scores
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        selected_feats = [feat for _, feat in sorted(zip(rfecv.ranking_, columns))[:rfecv.n_features_]]
        with open('selected_features_' + key + '.txt', 'w') as file_handle: 
            # file_handle.writelines(selected_feats)
            for feat in selected_feats: 
                file_handle.write("%s\n" % feat) 

        return rfecv, columns

    reg.fit(trainX, trainY)
    train_pred = reg.predict(trainX)
    test_pred = reg.predict(testX)
    print("Train mse: ", mean_squared_error(train_pred, trainY))
    print("Test mse: ", mean_squared_error(test_pred, testY))
    print("Train Score: ", reg.score(trainX, trainY))
    print("Test Score: ", reg.score(testX, testY))
    print("Pearson Correlation: ", pearson_coeff(test_pred, testY))