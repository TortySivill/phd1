
# x is a list of variables
# xll is a list of lower bounds
import numpy as np
import random


def split(list_a, chunk_size):

  for i in range(0, len(list_a), chunk_size):
    yield list_a[i:i + chunk_size]

dvs = {}

dvs[0] = [1,0]
dvs[1] = [1,0]
dvs[2] = [1,0]
dvs[3] = [1,0]

def interact(f,X1,X2,x,z,mini,maxi,e):
	temp_bounds = {}
	features = list(np.arange(len(x)))
	tick = 0
	for p in range(0,1):
		ref = [random.uniform(i,j) for i,j in zip(mini,maxi)]
		sample = [random.uniform(i,j) for i,j in zip(mini,maxi)]
		for i in features:
			if p == 0:
				temp_bounds[i] = ref[i]

		
		for i in X2:
			temp_bounds[i] = sample[i]

		for i in X1:
			temp_bounds[i] = sample[i]
		
		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		original_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]


		for i in X1:
			temp_bounds[i] = ref[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		new_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]
						

		sigma_1 = original_prediction - new_prediction

		for i in X2:
			temp_bounds[i] = ref[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		original_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]

		for i in X1:
			temp_bounds[i] = sample[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		new_prediction = f(np.asarray(temp).reshape(1,-1))[0][0]

		sigma_2 =  new_prediction - original_prediction 
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
			X11 = interact(f,tempX2,G1,x,z,mini,maxi,e)
			X21 = interact(f,tempX1,G2,x,z,mini,maxi,e)
			X1 = []
			"""if len(X11) == 1 and len(X21) == 1:
				for i in G1:
					if i not in X1:
						X1.append(i)
				for i in G2:
					if i not in X1:
						X1.append(i)
				for i in tempX1:
					if i not in X1:
						X1.append(i)"""
			#else:
			for i in X11:
				if i not in X1:
					X1.append(i)
			for i in X21:
					if i not in X1:
						X1.append(i)

	return X1

def RDG(f,x,z,mini,maxi,e):
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
		X1_star = interact(f,X1,X2,x,z,mini,maxi,e)
		
		if X1_star == X1:
			if len(X1) == 1:
				seps.append(X1)

			else:
				nonseps.append(X1)

			X1 = []
			X1_star = []
			X1.append(X2[0])
			X2.pop(0)
			comparison = list(np.arange(len(x)))
	
			for i in X1:
				comparison.remove(i)

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
			#if len(X2) == 0:

				#if len(X1) == len(features):

				#nonseps.append(X1)
		

	

	#seps.append(X1)
	temp_bounds = {}
	for i in np.arange(len(x)):
		temp_bounds[i] = 0


	atts = {}
	temp_bounds = {}

	
	for var in seps:

		for i in features:
			temp_bounds[i] = z[i]
		for i in var:
			temp_bounds[i] = x[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - f(np.asarray(z).reshape(1,-1))[0][0]

	for var in nonseps:
		

		for i in features:
			temp_bounds[i] = z[i]
		for i in var:
			temp_bounds[i] = x[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - f(np.asarray(z).reshape(1,-1))[0][0]

	
	return atts, seps, nonseps



def RDG_separate(f,x,z,mini,maxi,e):
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
		X1_star = interact(f,X1,X2,x,z,mini,maxi,e)
		
		if X1_star == X1:
			if len(X1) == 1:
				seps.append(X1)

			else:
				nonseps.append(X1)

			X1 = []
			X1_star = []
			X1.append(X2[0])
			X2.pop(0)
			comparison = list(np.arange(len(x)))
	
			for i in X1:
				comparison.remove(i)

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
			#if len(X2) == 0:

				#if len(X1) == len(features):

				#nonseps.append(X1)
		
	return seps,nonseps
	
def atts_separate(f,x,z,mini,maxi,e,seps,nonseps):
	features = list(np.arange(len(x)))
	#seps.append(X1)
	temp_bounds = {}
	for i in np.arange(len(x)):
		temp_bounds[i] = 0


	atts = {}
	temp_bounds = {}

	
	for var in seps:

		for i in features:
			temp_bounds[i] = z[i]
		for i in var:
			temp_bounds[i] = x[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - f(np.asarray(z).reshape(1,-1))[0][0]

	for var in nonseps:
		

		for i in features:
			temp_bounds[i] = z[i]
		for i in var:
			temp_bounds[i] = x[i]

		temp = []
		for i in features:
			temp.append(temp_bounds[i])

		atts[tuple(var)] = f(np.asarray(temp).reshape(1,-1))[0][0] - f(np.asarray(z).reshape(1,-1))[0][0]

	
	return atts