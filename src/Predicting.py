import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import json

BINS_N = 50

# TODO
# Test on Training, split it into two parts and do the schmilblik
# Compute metrics and evaluate models

def load_data(f_in="../data/training_treated.csv"):
    data = pd.DataFrame(pd.read_csv(f_in, sep=";", header=0))
    data = data.apply(pd.to_numeric, errors='ignore')
    return data

def training(target, model, data):
    X_cols = [4] + [x for x in range(7,data.shape[1])]
    X_train = data.iloc[:,X_cols]

    Y_train = data.loc[:,target]

    model.fit(X_train, Y_train)
    return model, model.score(X_train, Y_train), X_train, Y_train

def testing(model, data):
    X_test = data.iloc[:,:-5]
    return model.predict(X_test)

def error(Y_train, Y_predict):
    error=0
    for i in range(len(Y_train)):
        error+=(abs(Y_train[i]-Y_predict[i])/Y_train[i])
    train_error=error/len(Y_train)*100
    return train_error

targets = ["CapaciteEmprunt", "PrevisionnelAnnuel"]
models =   {"Linear": linear_model.LinearRegression(),
            "Ridge": linear_model.Ridge(),
            "BayesianRidge": linear_model.BayesianRidge(),
            "Huber": linear_model.HuberRegressor()}
scores = {}

all_data = load_data()
training_data = all_data
test_data = all_data

for t in targets:
    scores[t] = {}
    print("Target: "+t)
    for m in models:
        print("\tStarting "+m+"...")
        model, model_score, X_train, Y_train = training(t, models[m], data=training_data)
        scores[t][m] = model_score
        prediction = testing(model, data=test_data)

        plt.hist(prediction, bins=BINS_N, color="blue", log=True, density=True)
        plt.ylabel("Predicted " + str(t))
        plt.xlabel("Using "+str(m))
        plt.savefig("./images/"+str(t)+"_"+str(m)+".png")

with open("scores_prediction.json",'w+') as f_out:
    json.dump(scores,f_out, sort_keys=True, indent=4)