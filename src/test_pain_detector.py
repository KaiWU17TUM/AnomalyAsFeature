import os, sys

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath('.')))

from utils.vis import *
from utils.tsa import *

import pandas as pd

random_seed = 0
np.random.seed(random_seed)

from sklearn import linear_model, neural_network
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score, precision_score, recall_score, roc_auc_score

if __name__ == '__main__':
    df_data = pd.read_pickle(os.path.abspath('../data/data_pain_detector.pkl'))
    X_pos = df_data.loc[df_data['y']==1, 'X']
    X_neg = df_data.loc[df_data['y']==0, 'X']
    idx_subsample = np.random.choice(range(X_neg.shape[0]), X_pos.shape[0], replace=False)
    X = np.array(pd.concat((X_pos, X_neg.iloc[idx_subsample])).to_list())
    X = np.nan_to_num(X, nan=0.0)
    y = np.vstack((np.ones((X_pos.shape[0], 1)), np.zeros((X_pos.shape[0], 1))))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_seed)

    metrics = {
        'acc': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'auc': roc_auc_score,
    }


    model_lr = linear_model.LogisticRegression(max_iter=5000, random_state=random_seed)
    model_lr.fit(X_train, y_train)
    y_lr = model_lr.predict(X_test)

    lr_score = {
        k: metrics[k](y_test, y_lr) for k in metrics
    }


    model_mlp = neural_network.MLPClassifier(
        hidden_layer_sizes=(128, 256, 128, 64, 16),
        activation='relu',
        solver='adam',
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=3000,
        shuffle=True,
        early_stopping=True,
        validation_fraction=.1,
        random_state=random_seed
    )
    model_mlp.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())
    y_mlp = model_mlp.predict(X_test.reshape(X_test.shape[0], -1))

    mlp_score = {
        k: metrics[k](y_test, y_mlp) for k in metrics
    }




    print(11111)






