import numpy as np
import itertools
from itertools import permutations

def value_cond(f,S,S_bar,x,X,feature_value):
	

	S_datas_without = []
	for data in X:
		tip = 1
		for i in S:
			if data[i] == x[i]:
				continue
			else:
				tip = 0
				break

		if tip == 1:
			S_datas_without.append(data)
	
	S_datas_with = []
	for data in S_datas_without:
		if data[feature_value] == x[feature_value]:
			S_datas_with.append(data)


	val_with = f(np.asarray(S_datas_with))
	val_without = f(np.asarray(S_datas_without))

	return np.mean(val_with) - np.mean(val_without)

def value_cond_gaussian(f,S,barS,x,means,cov):
	

	S_value  = x[S]
	barS = np.asarray(barS)
	
	if len(S) > 0 and len(barS > 0):
		expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]
	
		temp_bounds = {}
		for i,j in zip(S,range(0,len(S))):
			temp_bounds[i] = S_value[j]
		for i,j in zip(barS,range(0,len(barS))):
			temp_bounds[i] = expected_S[j]

		temp = []
		for i in range(0,len(x)):
			temp.append(temp_bounds[i])
	elif len(S) > 0: 
		temp = x
	elif len(barS) > 0:
		temp = means

	else:
		print("oh dear")


	return f(np.asarray(temp).reshape(1,-1))

	#return np.mean(val_with) - np.mean(val_without)


def value_marg(f,S,S_bar,x,z,feature_value):
	xs = list(np.zeros(len(x)))
	if len(S) > 0:
		for i in S:
			xs[i] = x[i]
	if len(S_bar) > 0:
		for i in S_bar:
			xs[i] = z[i]

	with_i = xs.copy()
	with_i[feature_value] = x[feature_value]
	without_i = xs.copy()
	without_i[feature_value] = z[feature_value]
	
	return f(np.asarray(with_i).reshape(1,-1)) - f(np.asarray(without_i).reshape(1,-1))

def shapley_int(f,x,z):
	atts = {}
	n = x.shape[0]

	for feature_value in np.arange(x.shape[0]):
		perm_value = 0
		perms = 0
		for i in permutations(np.arange(n)):
			perms += 1
			S = []
			S_bar = []
			feature_index = i.index(feature_value)
			S = list(i[:feature_index])
			S_bar = list(i[feature_index+1:])
			
			perm_value += value_marg(f,S,S_bar,x,z,feature_value)



		atts[feature_value] = perm_value/perms
	return(atts)



def shapley_cond(f,x,X):
	atts = {}
	n = x.shape[0]

	for feature_value in np.arange(x.shape[0]):
		perm_value = 0
		perms = 0
		for i in permutations(np.arange(n)):
			perms += 1
			S = []
			S_bar = []
			feature_index = i.index(feature_value)
			S = list(i[:feature_index])
			S_bar = list(i[feature_index+1:])
			
			perm_value += value_cond(f,S,S_bar,x,X,feature_value)



		atts[feature_value] = perm_value/perms
	return(atts)

def shapley_cond_gaussian(f,x,X):
	atts = {}
	n = x.shape[0]

	cov = np.cov(X.T)
	means = X.mean(axis=0)

	for feature_value in np.arange(x.shape[0]):
		perm_value = 0
		perms = 0
		for i in permutations(np.arange(n)):
			perms += 1
			S = []
			S_bar = []
			feature_index = i.index(feature_value)
			S = list(i[:feature_index])
			S_bar = list(i[feature_index+1:])

			T = S.copy()
			T.append(feature_value)

			T_bar = S_bar.copy()
			T_bar.append(feature_value)
			perm_value += value_cond_gaussian(f,T,S_bar,x,means,cov) - value_cond_gaussian(f,S,T_bar,x,means,cov)


			#print(perm_value)
		atts[feature_value] = perm_value/perms
		#print(atts)
	return(atts)



