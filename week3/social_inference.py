#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import gamma

# utility functions

def power(l,y):
	"""
	Input: list of ints/floats or an individual int/float
	Output: list where each entry has been raised to the power y
	"""
	if isinstance(l,list):
		return np.power([float(x) for x in l],y)
	else:
		return np.power(float(l), y)

def pog(m1,v1,m2,v2):
	"""
	Product of two gaussians N(m1,v1), N(m2,v2)
	"""
	m_numer = (v1*m2) + (v2*m1)
	m_denom = (v1+v2)
	m = m_numer/m_denom
	
	v_denom = (1/v1) + (1/v2)
	v = 1/v_denom
	return m, v

def pog_multi(dists):
	"""
	Computes the product of multiple gaussians, following https://math.stackexchange.com/questions/1246358/the-product-of-multiple-univariate-gaussians.
	Input: a list with n tuples including mean and variance of univariate gaussians
	dists = [(m1,v1),(m2,v2)...(mn,vn)]
	"""
	means = []
	variances = []
	product_a = [] #for calculation of sigma_i**-2 * m_i
	product_b = [] #for calculation of sigma_i**-2 * m_i**2
	
	n = len(dists)
	for tup in dists:
		means.append(tup[0])
		variances.append(tup[1])
		product_a.append(power(tup[1],-2) * tup[0])
		product_b.append(power(tup[1],-2) * power(tup[0],2))
	
	var = power(np.sum(power(variances,-2)),(-1/2))
	mean = power(var,2) * np.sum(product_a)
	S_a = power((2*np.pi),((n-1)/2))
	S_b = var/(np.prod(variances))
	S_c = power(np.e,((0.5*power(var,-2)*power(mean,2)) - (0.5*np.sum(product_b))))
	S = S_a * S_b * S_c
	
	return mean, var, S
	
# plotting functions

def plot_gaussians(l):
	"""
	Plot n gaussians where the means and variances of the gaussians are specified as tuples within a list, l.
	l = [(m_1,v_1),(m_2,v_2)...(m_n,v_n)]
	Additionally, the tuples may have a third entry for the scaling factor of the gaussian.
	l = [(m_1,v_1,scale)...]
	"""
	means = []
	variances = []
	for tup in l:
		means.append(tup[0])
		variances.append(tup[1])
		
	x_min = min(means) - 3*max(variances)
	x_max = max(means) + 3*max(variances)
	x_axis = np.arange(x_min,x_max,1)
	
	for tup in l:
		try:
			y = [tup[2]*st.norm.pdf(x,tup[0],tup[1]) for x in x_axis]
		except IndexError:
			y = [st.norm.pdf(x,tup[0],tup[1]) for x in x_axis] # no scaling factor
		plt.plot(x_axis,y)
	plt.show()


# social category classes

class social_cat:
	"""
	A social category object that has various values, means, variances, and confidences in those values.
	An example call might be the following:
	gender = social_cat(('men','women'),(male_mean,female_mean),(male_var,female_var),(0.2,0.8))
	"""
	def __init__(self,values,means,variances,confidences):
		
		self.values = dict.fromkeys(values)
		
		# check for equal numbers of values, means, confs
		if (len(values) != len(means)) or (len(values) != len(confidences)) or (len(values) != len(variances)):
			raise ValueError('Warning! length of lists not equal!')
			
		# check that confs sum to 1
		if self.decimal_sum(confidences) != 1.0: # doesn't catch negative confidences
			raise ValueError('Warning! confidences do not sum to 1')
		
		# assign values
		for enum, value in enumerate(self.values):
			self.values[value] = {'mean': means[enum], 'variance': variances[enum], 'conf': confidences[enum]}
	
	def decimal_sum(self,l):
		"""
		Deal with issues/inaccuracies with floating point limitations
		"""
		return float("%0.1f" % float(sum(l)))
	
	def confidence_scale_pdf(self,array):
		pdf_values = []
		# marginalizing across values
		for value in self.values:
			scaled = [x * self.values[value]['conf'] for x in st.norm.pdf(array,self.values[value]['mean'],self.values[value]['variance'])]
			pdf_values.append(scaled)
		# element-wise sum of all scaled pdf values
		summed = [sum(x) for x in zip(*pdf_values)]
		self.scaled_pdf = summed



class social_categories:
	"""
	An object that is a collection of various social category objects, along with the confidence associated with each social category. We can interpret this as the listener's idea of the relevance of the category to the current task.
	An example call could be the following, assuming the existence of `gender` and `roo` social category objects already instantiated.
	current_set = social_categories((gender,roo),(0.0,1.0))
	"""

	def __init__(self,social_cats,confidences):
		self.values = dict(zip(social_cats,confidences))
	
	def confidence_scale_pdf(self,x):
		pdf_values = []
		# marginalizing across cats
		for cat in self.values.keys():
			scaled = [self.values[cat] * x for x in cat.scaled_pdf]
			pdf_values.append(scaled)

		# element-wise sum of all scaled pdf values
		summed = [sum(x) for x in zip(*pdf_values)]
		self.scaled_pdf = summed


# updating mean and variance

def pdf_scaled_inv_chi_sq(x,df,scale):
	"""
	Calculate the probability density of a scaled inverse chi-squared distribution.
	See wikipedia for details. 
	"""
	numer_1 = ((scale*df)/2)**(df/2)
	denom_1 = gamma(df/2)
	numer_2 = np.exp((-df*scale)/(2*x))
	denom_2 = x**(1 + (df/2))
	
	return (numer_1*numer_2)/(denom_1*denom_2)

def update_unknown_mean_var(mean_0,k_0,v_0,s2_0,obs):
	"""
	Gelman et al. (2014) p. 67-69
	Let X_1...X_n be a random sample (i.i.d.) from likelihood N(Mu, Sigma^2).
	The conjugate prior for this distribution can be broken down into the product of
		- Mu | Sigma^2 ~ N(Mu_0,Sigma^2/k_0)
		- Sigma^2 ~ Inv-X2(v_0,Sigma^2)
	The posterior distribution p(Mu, Sigma^2) is a Normal Inverse Chi-Squared Distribution with the following parameters (see p. 68):
		- Mu_n
		- k_n
		- v_n
		- v_n*(Sigma^2_n)
	"""
	# calculate sample parameters
	n = len(obs)
	mean_samp = np.mean(obs)
	ssd_samp = np.sum((obs - mean_samp)**2)/(n-1) # sum of squared deviations
	
	
	# updates for posterior parameters
	k_n = k_0 + n
	mean_n = ((k_0*mean_0)+(n*mean_samp))/k_n
	v_n = v_0 + n
	vnsn = (v_0*s2_0) + (n-1)*ssd_samp + (k_0*n*((mean_samp-mean_0)**2))/k_n
	s2_n = vnsn/v_n
	
	# marginal posterior density of Sigma^2 is a scaled inverse-X2 with parameters v_n and s2_n
	# expectation of scaled inv chi-squared is (v_n * s2_n) / (v_n - 2)
	sig_exp = (v_n*s2_n) / (v_n-2)
	
	return mean_n, sig_exp


# phonetic category classes

class phon_category:
	def __init__(self,label,mean,var):
		self.label = label
		self.mean = float(mean)
		self.var = float(var)
	def __repr__(self):
		return "category(" + ', '.join([self.label,str(self.mean),str(self.var)]) + ')'
	def __str__(self):
		return "category(" + ', '.join([self.label,str(self.mean),str(self.var)]) + ')'
	def pdf(self,value):
		return st.norm.pdf(value,self.mean,self.var)
	def sample(self,n_samples):
		samples = np.random.normal(self.mean,self.var,n_samples)
		return samples

def plot_cats(cat1,cat2,x_label):
	# Collect means and variances
	means = [c.mean for c in [cat1,cat2]]
	variances = [c.var for c in [cat1,cat2]]
	# Establish parameters and plot
	x_min = min(means) - 3*max(variances)
	x_max = max(means) + 3*max(variances)
	x_axis = np.arange(x_min,x_max,1)
	bound_vals = cat1.pdf(x_axis)/(cat1.pdf(x_axis)+cat2.pdf(x_axis))
	fig, (ax1,ax2) = plt.subplots(2,1)
	# Distribution plot
	ax1.plot(x_axis,cat1.pdf(x_axis))
	ax1.plot(x_axis,cat2.pdf(x_axis))
	ax1.set_title('Likelihood and Posterior Probability')
	ax1.set_ylabel('Probability Distribution')
	ax1.annotate(cat1.label, xy=(cat1.mean,cat1.pdf(cat1.mean)), xytext=(cat1.mean-100,cat1.pdf(cat1.mean)-(cat1.pdf(cat1.mean)/4)))
	ax1.annotate(cat2.label, xy=(cat2.mean,cat2.pdf(cat2.mean)), xytext=(cat2.mean-100,cat2.pdf(cat2.mean)-(cat2.pdf(cat2.mean)/4)))
	# Boundary plot
	ax2.plot(x_axis,bound_vals)
	ax2.set_xlabel(x_label)
	ax2.set_ylabel('Posterior Prob. "' + cat1.label + '"')
	plt.show()

def plot_multiple_cats(l):
	# Collect means and variances
	means = [c.mean for c in l]
	variances = [c.var for c in l]
	# Establish parameters and plot
	x_min = min(means) - 3*max(variances)
	x_max = max(means) + 3*max(variances)
	x_axis = np.arange(x_min,x_max,10)
	# Distribution plot
	for c in l:
		plt.plot(x_axis,c.pdf(x_axis))
		plt.annotate(c.label, xy=(c.mean,c.pdf(c.mean)), xytext=(c.mean+50,c.pdf(c.mean)+0.000025))
	
	plt.plot(x_axis,c.pdf(x_axis))
	plt.title('Category Distributions')
	plt.ylabel('Probability Distribution')
	plt.show()
