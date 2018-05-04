import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
import json

BINS_N = 50

def training(target, model, f_in="../data/training_treated.csv"):
    data = pd.DataFrame(pd.read_csv(f_in, sep=";", header=0))
    data = data.apply(pd.to_numeric, errors='ignore')

    X_cols = [4] + [x for x in range(7,data.shape[1])]
    X_train = data.iloc[:,X_cols]

    Y_train = data.loc[:,target]

    model.fit(X_train, Y_train)
    return model, model.score(X_train, Y_train)

def testing(model, f_in="../data/test_treated.csv"):
    data = pd.DataFrame(pd.read_csv(f_in, sep=";", header=0))
    data = data.apply(pd.to_numeric, errors='ignore')
    X_test = data.iloc[:,:-5]
    return(model.predict(X_test))

targets = ["CapaciteEmprunt", "PrevisionnelAnnuel"]
models =   {"Linear": linear_model.LinearRegression(),
            "Ridge": linear_model.Ridge(),
            "BayesianRidge": linear_model.BayesianRidge(),
            "Huber": linear_model.HuberRegressor()}
scores = {}

for t in targets:
    scores[t] = {}
    print("Target: "+t)
    for m in models:
        print("\tStarting "+m+"...")
        model, model_score = training(t, models[m])
        scores[t][m] = model_score
        prediction = testing(model)

        plt.hist(prediction, bins=BINS_N, color="blue", log=True, density=True)
        plt.ylabel("Predicted " + str(t))
        plt.xlabel("Using "+str(m))
        plt.savefig("./images/"+str(t)+"_"+str(m)+".png")

with open("scores_prediction.json",'w+') as f_out:
    json.dump(scores,f_out, sort_keys=True, indent=4)