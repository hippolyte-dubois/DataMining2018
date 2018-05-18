import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from sklearn import linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt
import json, io, csv
import seaborn as sns

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
    
    f, ax = plt.subplots(figsize=(30,30))
    ax = sns.heatmap(X_train.corr(), mask=np.zeros_like(X_train.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
    f.savefig("Correlation_heatmap.png")

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

def treatment(t, scores, models, all_data):
    scores[t] = {}
    best_model = None
    best_score = 0
    training_data = all_data.sample(frac=0.5)
    test_data = all_data[~all_data.isin(training_data)].dropna()
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
        plt.title("Using "+m+". Score: "+str(model_score))
        plt.subplot(1,2,1)
        plt.hist(prediction[0], bins=BINS_N, color="blue", log=LOG, density=True)
        plt.ylabel("Predicted " + str(t))
        plt.xlabel("Values (histogram)")
        #print("-"*20 + "\n" + t + ": " + m, file=output_f)
        #for i in range(len(prediction[0])):
        #    print(prediction[1][i] + ": " + str(prediction[0][i]), file=output_f)

        error_rate = error(np.array(test_data[t]), prediction[0]) 

        plt.subplot(1,2,2)
        plt.hist(error_rate, bins=BINS_N, color="red", log=LOG, density=True)
        plt.xlabel("Distance to the real value (histogram)")
        plt.savefig("./images/"+str(t)+"_"+str(m)+".png")
        plt.clf()
        plt.cla()
    plt.hist(test_data[t], bins=BINS_N, color="blue", log=LOG, density=True)
    plt.ylabel("Actual values")
    plt.xlabel("Values of " + t + " (histogram)")
    plt.savefig("./images/"+str(t)+"_Actual.png")
    plt.clf()
    plt.cla()
    return best_model

def main():
    models =   {"Linear": linear_model.LinearRegression(),
                "Ridge": linear_model.Ridge(),
                "BayesianRidge": linear_model.BayesianRidge(),
                "Huber": linear_model.HuberRegressor()}
    scores = {}
    all_data = load_data()

    # On sélectionne le meilleur modèle, on le rentraine avec toutes les données cette fois ci
    model_cap_emprunt = treatment("CapaciteEmprunt", scores, models, all_data)
    model_cap_emprunt, cap_score, X_cap, Y_cap = training("CapaciteEmprunt", model_cap_emprunt, all_data)
    model_prev_annuel = treatment("PrevisionnelAnnuel", scores, models, all_data)
    model_prev_annuel, pre_score, X_pre, Y_pre = training("PrevisionnelAnnuel", model_prev_annuel, all_data)

    with open("scores_prediction.json",'w+') as f_out:
        json.dump(scores,f_out, sort_keys=True, indent=4)

    with open("../data/test_treated.csv","r") as in_f:
        out_f = open("../data/test_predicted.csv", "w+")
        reader = csv.reader(in_f, delimiter=';')
        writer = csv.writer(out_f, delimiter=";")
        header = reader.__next__()
        writer.writerow(header)
        for row in reader:
            X = np.array(row[:31], dtype=np.float32).reshape(1, -1)
            try:
                prev_annuel = model_prev_annuel.predict(X)
                cap_emprunt = model_cap_emprunt.predict(X)
            except ValueError as e:
                print("Error. Line:")
                print(row)
                print(e)
            else:
                out = list(X[0])
                out.append(float(cap_emprunt[0]))
                out.append(float(prev_annuel[0]))
                out.append(0)
                out.append(0)
                out.append(0)
                writer.writerow(out)
        out_f.close()

main()