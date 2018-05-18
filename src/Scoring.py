import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import json, io, csv

BINS_N = 50
LOG=True

def load_data(f_in="../data/training_treated.csv"):
    data = pd.DataFrame(pd.read_csv(f_in, sep=";", header=0))
    data = data.apply(pd.to_numeric, errors='ignore')
    #Recupère uniquement les lignes qui ont un secteur (col 1, 2 ou 3 == 1)
    data = data.loc[(data.Secteur1) | (data.Secteur2) | (data.SecteurParticulier)]
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

models = {"LinearSVC": svm.LinearSVC(), "Neighbors": neighbors.KNeighborsClassifier()}
targets = ["Secteur1", "Secteur2", "SecteurParticulier"]

scores = {}
best_models = {}

#Choper le meilleur model (et rajouter des models)


all_data = load_data()
training_data = all_data.sample(frac=0.5)
test_data = all_data[~all_data.isin(training_data)].dropna()

for t in targets:
	scores[t] = {}
	best_model = None
	best_score = 0
	print("Target: "+t)
	for m in models:
	    print("Model : "+m)
	    scores[t][m] = {}
	    model, model_score, X_train, Y_train = training(t, models[m], data=training_data)
	    if model_score > best_score:
        	best_model = model
        	best_score = model_score
	    scores[t][m]["score"] = model_score
	    prediction = testing(model, data=test_data)

	    plt.subplot(1,2,1)
	    plt.hist(prediction[0], bins=BINS_N, color="blue", log=LOG, density=True)
	    plt.ylabel("Nb of values")
	    plt.xlabel("Predicted "+str(t))

	    plt.subplot(1,2,2)
	    plt.hist(np.array(test_data[t]), bins=BINS_N, color="red", log=LOG, density=True)
	    plt.xlabel("Real "+str(t))

	    plt.savefig("./images/"+str(t)+"_"+str(m)+".png")
	    plt.clf()
	    plt.cla()

	    #Matrice de confusion
	    Y_pred = prediction[0]
	    Y_true = test_data[t]
	    confusion_matrix = ConfusionMatrix(Y_true, Y_pred)
	    print("Confusion matrix:\n%s" % confusion_matrix)

	    #Aucune idée de quoi faire de ces résultats, mais ils sont là
	    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_true, Y_pred))  
	    print('Mean Squared Error:', metrics.mean_squared_error(Y_true, Y_pred))  
	    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_true, Y_pred)))
	    print("Recall: "+str(metrics.recall_score(Y_true,Y_pred, average='binary')))
	    print("Precision: "+str(metrics.precision_score(Y_true,Y_pred, average='binary')))
	    accuracy = metrics.accuracy_score(Y_true,Y_pred)
	    print("Accuracy: "+str(accuracy))
	    print("Error Rate: "+str(1 - accuracy))
	best_models[t] = best_model
		

with open("scores_prediction_sectors.json",'w+') as f_out:
    json.dump(scores,f_out, sort_keys=True, indent=4)


#Appliquer les meilleurs models pour le set final
with open("../data/test_predicted.csv","r") as in_f:
    out_f = open("../data/test_predicted_final.csv", "w+")
    reader = csv.reader(in_f, delimiter=';')
    writer = csv.writer(out_f, delimiter=";")
    header = reader.__next__()
    writer.writerow(header)
    for row in reader:
        X = np.array(row[:31], dtype=np.float16).reshape(1, -1)
        try:
            Secteur1 = best_models["Secteur1"].predict(X)
            Secteur2 = best_models["Secteur2"].predict(X)
            SecteurParticulier = best_models["SecteurParticulier"].predict(X)
        except ValueError:
            print(X)
        else:
            out = list(X[0])
            out.append(float(row[31]))
            out.append(float(row[32]))
            out.append(float(Secteur1[0]))
            out.append(float(Secteur2[0]))
            out.append(float(SecteurParticulier[0]))
            writer.writerow(out)
    out_f.close()
