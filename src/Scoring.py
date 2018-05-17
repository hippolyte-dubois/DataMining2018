import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import json

BINS_N = 50
LOG=True

def load_data(f_in="../data/training_treated.csv"):
    data = pd.DataFrame(pd.read_csv(f_in, sep=";", header=0))
    data = data.apply(pd.to_numeric, errors='ignore')
    #Recupère uniquement les lignes qui ont un secteur (col 1, 2 ou 3 == 1)
    #data = data.loc[(data.Secteur1 == 1) | (data.Secteur2 == 1) | (data.SecteurParticulier == 1)]
    data = data.loc[(data.Secteur > 0)]
    print(data)
    return data

def training(model, data):
    X_cols = [4] + [x for x in range(7,data.shape[1])]
    X_train = data.iloc[:,X_cols]

    #Y_cols = ["Secteur1","Secteur2","SecteurParticulier"]
    Y_cols = ["Secteur"]

    #https://stackoverflow.com/questions/31306390/sklearn-classifier-get-valueerror-bad-input-shape
    
    Y_train = data.loc[:,Y_cols]

    model.fit(X_train, Y_train)
    return model, model.score(X_train, Y_train), X_train, Y_train

def testing(model, data):
    X_cols = [4] + [x for x in range(7,data.shape[1])]
    X_test = data.iloc[:,X_cols]
    X_ID = data.loc[:,"Client"]
    return model.predict(X_test), np.array(X_ID)

models = {"SVC": svm.SVC()}

scores = {}


all_data = load_data()
training_data = all_data.sample(frac=0.5)
test_data = all_data[~all_data.isin(training_data)].dropna()

for m in models:
    print("Model : "+m)
    scores[m] = {}
    model, model_score, X_train, Y_train = training(models[m], data=training_data)
    scores[m]["score"] = model_score
    prediction = testing(model, data=test_data)

    print(scores)
    print(prediction[1])
    #TODO print le bordel
    plt.subplot(1,2,1)
    plt.hist(prediction[0], bins=BINS_N, color="blue", log=LOG, density=True)
    plt.ylabel("Nb of values")
    plt.xlabel("Predicted Secteur")

    plt.subplot(1,2,2)
    plt.hist(np.array(test_data["Secteur"]), bins=BINS_N, color="red", log=LOG, density=True)
    plt.xlabel("Real Secteur")

    plt.savefig("./images/Secteur_"+str(m)+".png")
    plt.clf()
    plt.cla()

    #TODO faire la matrice de confusion

    #TODO faire les calculs d'erreurs (les 3) et accuration, precision et tout ce bordel

#TODO 