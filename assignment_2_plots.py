# Assignment 2: Breast cancer diagnosis using a Support Vector Machine
 
# https://uppsala.instructure.com/courses/65352/assignments/154750?module_item_id=627882


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection, svm
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def gradient(lmb, w, b, sign, y, x):
	return np.array([lmb*w, 0]) if sign else np.array([(lmb*w-y*x),-y])

def constraint(x, y, w, b):
	return y*(np.dot(w,x)+b) >= 1

def data():

	cols = ["ID", "Diag", "r_m", "txt_m", "per_m", "area_m", "smth_m", "comp_m", "conc_m", "conc_p_m", "sym_m", "frac_m", 
			"r_s", "txt_s", "per_s", "area_s", "smth_s", "comp_s", "conc_s", "conc_p_s", "sym_s", "frac_s",
			"r_w", "txt_w", "per_w", "area_w", "smth_w", "comp_w", "conc_w", "conc_p_w", "sym_w", "frac_w"]

	data = pd.read_table("wdbc.dat", sep=",", usecols = list(range(32)), names = cols)
	data["Diag"] = data["Diag"].replace(['M', 'B'], [1, -1])

	X = data.loc[:, ~data.columns.isin(['ID', 'Diag'])]
	y = data["Diag"]


	X = X.to_numpy()
	y = y.to_numpy()

	return X, y

def traintest(X, y):
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, shuffle=True, random_state=1)

	return X_train, X_test, y_train, y_test


def SVM(X, y, lmb=0.01, gamma=0.001):

	#maxiter = 100
	w = np.zeros(X.shape[1])
	b = 0

	for _ in range(maxiter): 
		for i, x in enumerate(X): 

			D = gradient(lmb, w, b, constraint(X[i], y[i], w, b), y[i], X[i])

			w -= gamma*D[0]
			b -= gamma*D[1]

	return w, b

def statistics(y_true, y_pred): # 1 är positiv -1 är negativ (positiv = cancer)
	#Accuracy - Proportion of correct predictions - (TP+TN)/(TP+FP+TN+FN) = (TP+TN)/(ALL)
	accuracy = np.sum(y_true==y_pred)/len(y_true)
	#Sensitivity - proportion of positive diagnoses for patients with malignant cells - (TP)/(TP+FN) = (TP)/(P)
	sensitivity = np.sum(np.logical_and(y_true == 1, y_pred == 1))/np.sum(y_true == 1)
	#Specificity - proportion of negative diagnoses for patients without the disease - (TN)/(TN+FP) = (TN)/(N)
	specificity = np.sum(np.logical_and(y_true == -1, y_pred == -1))/np.sum(y_true == -1)
	return accuracy, sensitivity, specificity


def main():
	global maxiter
	maxiters = [2000]
	X, y = data()
	X_train, X_test, y_train, y_test = traintest(X, y)

	lambdas = [1, 0.1, 0.01, 0.001, 0.0001]

	for m in maxiters:
		maxiter = m
		statres = []
		for i in range(len(lambdas)):

			w,b = SVM(X_train,y_train,lambdas[i])
			est = np.dot(X_test, w) + b
			result = np.sign(est)
			#prediction = np.sign(est)
			#result = np.where(prediction == -1, -1, 1)
			stats = statistics(y_test, result)
			statres.append(stats)
			print("SVM Accuracy: ", stats[0])
			print("SVM Sensitivity: ", stats[1])
			print("SVM Specificity: ", stats[2])
		
		GM = svm.SVC(kernel = 'linear')
		GM.fit(X_train,y_train)
		statsSK = statistics(y_test, GM.predict(X_test))
		print("SKlearn Accuracy: ", statsSK[0])
		print("SKlearn Sensitivity: ", statsSK[1])
		print("SKlearn Specificity: ", statsSK[2])

		statres = np.array(statres)

		plt.plot(lambdas,statres[:,0])
		plt.plot(lambdas,statres[:,1])
		plt.plot(lambdas,statres[:,2])
		plt.title(f"maxiter = {m}")
		plt.xscale("log")
		plt.xlabel("lambda")
		plt.ylabel("%")
		plt.legend(['Accuracy','Sensitivity','Specificity'])
		#plt.show()

		filename = f"plot_{m}.png"
		plt.savefig(filename)
		plt.clf()

	return

main()
