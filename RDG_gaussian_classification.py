
# x is a list of variables
# xll is a list of lower bounds
import numpy as np
import random


def split(list_a, chunk_size):

  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]




def closest_value(input_list, input_value):
 
  difference = lambda input_list : abs(input_list - input_value)

 
  res = min(input_list, key=difference)
 
  return res

def find_sample(lb):



	minimum = 100
	minimum_data = []
	for data in datas:
		temp = np.linalg.norm(np.asarray(lb) - np.asarray(data))
		if temp < minimum:
			minimum = temp
			minimum_data = data

	
	return minimum_data


def interact(f,X1,X2,x,means,cov,mini,maxi,e,X):
	
	temp_bounds = {}

	features = list(np.arange(len(list(x))))


	for i in features:
	
		temp_bounds[i] = 0

	tick = 0	
	for p in range(0,1):
		ref = X[random.randint(0,X.shape[0]-1)]
		S = []
		barS = []
		for i in features:
			if (i in X2) or (i in X1):
				S.append(i)
			else:
				barS.append(i)

		S_value  = ref[S]
		barS = np.asarray(barS)


		if len(barS) > 0:
			expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]

			for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[j]
			for i,j in zip(barS,range(0,len(barS))):
				temp_bounds[i] = expected_S[j]
		else:
			for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[i]


		temp = []
		for i in features:
			temp.append(temp_bounds[i])
		
		original_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]
			

		S = []
		barS = []
		for i in features:
			if (i in X2):
				S.append(i)
			else:
				barS.append(i)

		S_value  = ref[S]
		barS = np.asarray(barS)
		S = np.asarray(S)
		S_value = np.asarray(S_value)

		

		expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]
		for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[j]
		for i,j in zip(barS,range(0,len(barS))):
			temp_bounds[i] = expected_S[j]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])
		
		new_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]

		sigma_1 = original_prediction - new_prediction


		original_prediction = f(np.asarray(means).reshape(1,-1))[0][0]

		S = []
		barS = []
		for i in features:
			if (i in X1):
				S.append(i)
			else:
				barS.append(i)

		S_value  = ref[S]
		barS = np.asarray(barS)
		

		expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]


		for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[j]
		for i,j in zip(barS,range(0,len(barS))):
			temp_bounds[i] = expected_S[j]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])
		
		new_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]

		sigma_2 = new_prediction - original_prediction




		if np.abs(sigma_1-sigma_2) > e:
			tick = 1

	if tick == 1:

		if len(X2) == 1:
			X1.append(X2[0])
			
			#X1 = [item for sublist in X1 for item in sublist]
		else:
			#two_lists = list(split(X2,int(len(X2)/2)))
			G1 = X2[:int(len(X2)/2)]
			G2 = X2[int(len(X2)/2):]
			tempX1 = X1.copy()
			tempX2 = X1.copy()
			X11 = interact(f,tempX2,G1,x,means,cov,mini,maxi,e,X)
			X21 = interact(f,tempX1,G2,x,means,cov,maxi,maxi,e,X)
			X1 = []	
			for i in X11:
				if i not in X1:
					X1.append(i)
			for i in X21:
				if i not in X1:
					X1.append(i)
			
			

	return X1

def RDG(f,x,means,cov,mini,maxi,e,X):
	seps = []
	nonseps = []
	#xll = [dvs[0][0],dvs[1][2],dvs[2][2],dvs[3][2]]
	#yll = f(xll)
	features = list(np.arange(len(x)))
	X1 = []
	X2 = []
	X1.append(features[0])
	X2 = features[1:]
	comparison = X2
	while len(X2) != 0:
		X1_star = interact(f,X1,X2,x,means,cov,mini,maxi,e,X)
		
		if X1_star == X1:
			if len(X1) == 1:
				seps.append(X1)

			else:
				nonseps.append(X1)

			X1 = []
			X1_star = []
			X1.append(X2[0])	
			X2.pop(0)

			tick = 0
			if len(X2) == 0:
				if len(X1) == 1:
					for u in nonseps:
						if X1[0] in u:
							tick = 1
							break
					if tick == 0:
						seps.append(X1)

		
		else:
			X1 = X1_star
			for i in X1:
				if i in X2:
					X2.remove(i)
					#continue
			if len(X2) == 0:

				#if len(X1) == len(features):

				nonseps.append(X1)
		

	

	#seps.append(X1)
	"""temp_bounds = {}
	for i in np.arange(len(x)):
		temp_bounds[i] = 0"""


	atts = {}
	temp_bounds = {}

	f_expected = f(np.asarray(means).reshape(1,-1))[0][0]

	if len(seps) > 0:
		for var in seps:
				S = []
				barS = []
				for i in features:
					if (i in var):
						S.append(i)
					else:
						barS.append(i)

				S_value  = x[S]
				barS = np.asarray(barS)
				expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]

				for i,j in zip(S,range(0,len(S))):
					temp_bounds[i] = S_value[j]
				for i,j in zip(barS,range(0,len(barS))):
					temp_bounds[i] = expected_S[j]

				temp = []
				for i in features:
					temp.append(temp_bounds[i])
				
				atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - np.mean(f(X))[0][0]

	for var in nonseps:
		S = []
		barS = []
		for i in features:
			if (i in var):
				S.append(i)
			else:
				barS.append(i)

		S_value  = x[S]


		if len(S) > 0 and len(barS) > 0:
			print("in here")

			barS = np.asarray(barS)
			S = np.asarray(S)
			S_value = np.asarray(S_value)
			expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]

			for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[j]
			for i,j in zip(barS,range(0,len(barS))):
				temp_bounds[i] = expected_S[j]

			print()
			temp = []
			for i in features:
				temp.append(temp_bounds[i])

			atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - np.mean(f(X))[0][0]

		else:
			if len(barS) > 0:
				atts[str(var)] = np.mean(f(X))[0][0]
			elif len(S) > 0:
				atts[tuple(var)] = f(np.asarray(x).reshape(1,-1))[0][0][0] - np.mean(f(X))[0][0]

		
		
		

		
	return atts, seps, nonseps



"""def f(xs):
	return sum(xs)


datas = []
for i in range(10000):
	data = []
	pop = random.randint(0,3)
	data.append(pop)
	data.append(pop)
	for i in range(0,97):
		data.append(random.randint(0,3))
	data.append(pop)

	datas.append(data)

datas.append(list(np.ones(100)))
datas = np.asarray(datas)

cov = np.cov(datas.T)
means = datas.mean(axis=0)

x = np.ones(100)


atts, seps, nonseps = RDG(f, x, means,cov)

print(atts)

print(seps)
print(nonseps)"""





def RDG_separate(f,x,means,cov,mini,maxi,e,X):
	seps = []
	nonseps = []
	#xll = [dvs[0][0],dvs[1][2],dvs[2][2],dvs[3][2]]
	#yll = f(xll)
	features = list(np.arange(len(x)))
	X1 = []
	X2 = []
	X1.append(features[0])
	X2 = features[1:]
	comparison = X2
	while len(X2) != 0:
		X1_star = interact(f,X1,X2,x,means,cov,mini,maxi,e,X)
		
		if X1_star == X1:
			if len(X1) == 1:
				seps.append(X1)

			else:
				nonseps.append(X1)

			X1 = []
			X1_star = []
			X1.append(X2[0])	
			X2.pop(0)

			tick = 0
			if len(X2) == 0:
				if len(X1) == 1:
					for u in nonseps:
						if X1[0] in u:
							tick = 1
							break
					if tick == 0:
						seps.append(X1)

		
		else:
			X1 = X1_star
			for i in X1:
				if i in X2:
					X2.remove(i)
					#continue
			if len(X2) == 0:

				#if len(X1) == len(features):

				nonseps.append(X1)
	return seps,nonseps





def atts_separate(f,x,means,cov,mini,maxi,e,X,seps,nonseps):
	features = list(np.arange(len(x)))
	atts = {}
	temp_bounds = {}

	f_expected = f(np.asarray(means).reshape(1,-1))[0][0]

	if len(seps) > 0:
		for var in seps:
				S = []
				barS = []
				for i in features:
					if (i in var):
						S.append(i)
					else:
						barS.append(i)

				S_value  = x[S]
				barS = np.asarray(barS)
				expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]

				for i,j in zip(S,range(0,len(S))):
					temp_bounds[i] = S_value[j]
				for i,j in zip(barS,range(0,len(barS))):
					temp_bounds[i] = expected_S[j]

				temp = []
				for i in features:
					temp.append(temp_bounds[i])
				
				atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - np.mean(f(X))

	print("here")
	for var in nonseps:
		S = []
		barS = []
		for i in features:
			if (i in var):
				S.append(i)
			else:
				barS.append(i)

		S_value  = x[S]


		if len(S) > 0 and len(barS) > 0:
			print("in here")

			barS = np.asarray(barS)
			S = np.asarray(S)
			S_value = np.asarray(S_value)
			expected_S = (means[barS.reshape(barS.shape[0],1)] + (cov[barS.reshape(barS.shape[0],1),S]*cov[S,S])*(S_value-means[S]).reshape(len(S),1).T)[:,0]

			for i,j in zip(S,range(0,len(S))):
				temp_bounds[i] = S_value[j]
			for i,j in zip(barS,range(0,len(barS))):
				temp_bounds[i] = expected_S[j]

			print()
			temp = []
			for i in features:
				temp.append(temp_bounds[i])

			atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - np.mean(f(X))

		else:
			if len(barS) > 0:
				atts[str(var)] = np.mean(f(X))[0][0]
			elif len(S) > 0:
				atts[tuple(var)] = f(np.asarray(x).reshape(1,-1))[0][0] - np.mean(f(X))

		
	return atts



