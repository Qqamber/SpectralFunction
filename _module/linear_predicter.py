### Linear Prediction

import numpy as np

class LinearPrediction():
	'''
	extrapolate a time series of data(equidistant in time) of space-time correlators
	using linear prediction
	Ref:
	1: Spectral functions in one-dimensional quantum systems at T>0 [arXiv:0901.2342v2]
	2: The density-matrix renormalization group in the age of matrix product states [arXiv:1008.3477v2] Section 8.2
	'''
	def __init__(self, order, start, end, cutoff):
		'''initialize the parameters, index obeys python's convention
		Args:
		order: the number of previous datas used to predict the current point, [int type]
		[start, end]: specify the data set used to minimize the least square error, [int type]
		cutoff: argument for pseudo-inverse
		'''
		if start < order:
			print("ATTENTION: start can not be smaller than order, please redefine your model!")

		self.p = order
		self.start = start
		self.end = end
		self.a = np.zeros([self.p, 1]) # model parameters
		self.cutoff = cutoff


	def extrapolate(self, datas, num):
		'''main extrapolation process
		Args:
		datas: data used for extrapolation [list type]
		num: number of points to be extrapolated
		'''
		self._train(datas)

		para = (self.a).flatten().tolist()
		para.reverse()

		data = datas.copy()

		for i in range(num):
			x = data[-self.p:]
			new = - np.sum(np.multiply(x, para))
			data.append(new)

		return data[-num:]


	def _train(self, datas):
		'''train a linear model for extrapolation based on given data
		Args:
		datas: data used for traning, [list type]
		'''
		if len(datas) < self.end+1:
			print("Your data set is too small, please redefine your model or generate more data!")
			return 0
		elif len(datas) > self.end+1:
			confirm = input("Your data set is too big, and some part of it will be useless.\
				\nBe sure that you want this!(y/n)")
			if confirm != 'y':
				return 0

		R = np.zeros([self.p, self.p]).astype(complex)
		r = np.zeros([self.p, 1]).astype(complex)

		for j in range(self.p):
			for m in range(self.start, self.end+1):
				r[j] += np.conj(datas[m-j-1]) * datas[m]
			for i in range(self.p):
				for k in range(self.start, self.end+1):
					R[j, i] += np.conj(datas[k-j-1]) * datas[k-i-1]

		a = - np.dot(np.linalg.pinv(R, rcond=self.cutoff), r) # cutoff should be small

		# regularization, see Ref 1
		for i in range(self.p):
			if abs(a[i,0]) >= 1.:
				a[i,0] = 0.+0j

		self.a = a