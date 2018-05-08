import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt
import json, io, csv

BINS_N = 50
LOG=True

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
    X_cols = [4] + [x for x in range(7,data.shape[1])]
    X_test = data.iloc[:,X_cols]
    X_ID = data.loc[:,"Client"]
    return model.predict(X_test), np.array(X_ID)

def error(Y_train, Y_predict):
    error=[]
    for i in range(len(Y_train)):
        error.append(abs(Y_train[i] - Y_predict[i]))
    return error

"""
targets = ["CapaciteEmprunt", "PrevisionnelAnnuel"]
models =   {"Linear": linear_model.LinearRegression(),
            "Ridge": linear_model.Ridge(),
            "BayesianRidge": linear_model.BayesianRidge(),
            "Huber": linear_model.HuberRegressor()}
scores = {}

all_data = load_data()
training_data = all_data.sample(frac=0.5)
test_data = all_data[~all_data.isin(training_data)].dropna()

output_f = open("predicted.txt","a")

best_model = None
best_score = 0

for t in targets:
    scores[t] = {}
    print("Target: "+t)
    for m in models:
        scores[t][m] = {}
        print("\tStarting "+m+"...")
        model, model_score, X_train, Y_train = training(t, models[m], data=training_data)
        if model_score > best_score:
            best_model = model
            best_score = model_score
        scores[t][m]["score"] = model_score
        prediction = testing(model, data=test_data)

        #plt.tight_layout()
        plt.subplot(1,2,1)
        plt.hist(prediction[0], bins=BINS_N, color="blue", log=LOG, density=True)
        plt.ylabel("Predicted " + str(t))
        plt.xlabel("Values (histogram)")
        #print("-"*20 + "\n" + t + ": " + m, file=output_f)
        #for i in range(len(prediction[0])):
        #    print(prediction[1][i] + ": " + str(prediction[0][i]), file=output_f)

        error_rate = error(np.array(test_data[t]), prediction[0])
        scores[t][m]["error"] = error_rate

        plt.subplot(1,2,2)
        plt.hist(error_rate, bins=BINS_N, color="red", log=LOG, density=True)
        plt.xlabel("Distance to the real value (histogram)")
        plt.savefig("./images/"+str(t)+"_"+str(m)+".png")
        plt.clf()
        plt.cla()

with open("scores_prediction.json",'w+') as f_out:
    json.dump(scores,f_out, sort_keys=True, indent=4)
"""
with open("../data/test_treated.csv","r") as in_f:
    reader = csv.DictReader(in_f)
    for row in reader:
        X = 