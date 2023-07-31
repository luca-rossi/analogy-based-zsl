import numpy as np
import random
import torch

class SimilarSampleFinder:
	'''
	The idea is to find the most similar class to the current one (later it could be more than one, but let's keep it simple for now),
	sample a feature vector from that class and use it to condition the GAN.
	There are two parts to this:
	- Find the most similar seen class to the current one. This is done only once at the beginning.
	- Sample from the most similar seen class. This is done whenever we need to condition the GAN. This needs to be efficient.
	'''

	def __init__(self, data):
		'''
		Here, we find the most similar seen classes for each class.
		We simply define a matrix of labels, where each row corresponds to a class and each column corresponds to the index of a seen class sorted by similarity.
		The similarity is measured as the distance between the signatures (no features are needed, only attributes).
		'''
		self.data = data
		self.n_classes = data.get_n_classes()
		# create a two-dimensional array of size n_classes x n_seen_classes, where each element is the index of a seen class sorted by similarity
		similarities = np.zeros((self.n_classes, len(data.seen_classes)))
		# for each class...
		for i in range(self.n_classes):
			# get the class signature
			signature = data.attributes[i]
			# now we need to compare this signature to all the signatures of the seen classes (excluding the current one)
			# we can do this by simply subtracting the current signature from all the signatures of the seen classes
			# and then finding the norms and sorting them
			norms = []
			indexes = []
			for j in data.seen_classes:
				if j == i:
					# we can't use the same class as the most similar one, so we set the norm to infinity
					comparison_norm = np.inf
				else:
					other_signature = data.attributes[j]
					comparison = signature - other_signature
					comparison_norm = np.linalg.norm(comparison)
				indexes.append(j)
				norms.append(comparison_norm)
			# sort the norms and indexes in ascending order
			sorted_norms, sorted_indexes = zip(*sorted(zip(norms, indexes)))
			# store the sorted indexes in the similarities matrix
			similarities[i] = sorted_indexes
		self.similarities = similarities
		print(similarities)

	def get_sample(self, label, n_features, noise_size, cond_size, k=1, agg_type='concat', pool_type='mean'):
		'''
		Given a label, a number of similar classes to use (default 1), and the type of aggregation (concat or mean) and pooling (mean, max, or first),
		returns a feature vector from the most similar seen class or a fused feature vector from the most similar seen classes.
		'''
		# generate a noise vector
		noise = torch.randn(noise_size)
		# return the noise vector if we don't want to condition the GAN on similar classes features
		if k == 0:
			return noise
		# let's make the length of every feature vector equal cond_size
		pooling_size = n_features // cond_size
		# get the k most similar seen classes
		similar_labels = self.similarities[label][:k] # use the first k columns of the similarities matrix
		# initialize an empty list of feature vectors
		feature_vectors = []
		# loop over the k most similar seen classes
		for similar_label in similar_labels:
			# get the locations of the occurrencies of the label provided as input
			locations = np.where(self.data.train_Y == similar_label)[0]
			# from these locations, we need to sample a random one
			location = random.choice(locations)
			# now get the feature vector at that location
			feature_vector = self.data.train_X[location]
			# apply pooling to reduce dimensionality
			feature_vector = feature_vector.view(-1, pooling_size)
			if pool_type == 'mean':
				feature_vector = torch.mean(feature_vector, dim=1)
			elif pool_type == 'max':
				feature_vector = torch.max(feature_vector, dim=1).values
			elif pool_type == 'first':
				feature_vector = feature_vector[:, 0]
			else:
				raise ValueError('Invalid pooling type')
			# append the feature vector to the list
			feature_vectors.append(feature_vector)
		# check the flag for feature fusion
		if agg_type == 'concat':
			# stack the feature vectors along the dimension 0
			feature_vector = torch.stack(feature_vectors, dim=0)
			# flatten the feature vector
			feature_vector = feature_vector.flatten()
		elif agg_type == 'mean':
			# average the feature vectors along the dimension 0
			feature_vector = torch.mean(torch.stack(feature_vectors, dim=0), dim=0)
		else:
			raise ValueError('Invalid aggregation type.')
		# stack the noise vector to the feature vector
		feature_vector = torch.cat((feature_vector, noise), 0)
		return feature_vector

	def get_samples(self, labels, n_features, noise_size, cond_size, k=1, agg_type='concat', pool_type='mean'):
		'''
		Given a batch of labels, returns a batch of features from the most similar seen classes.
		'''
		# call the get_sample function for each label (make it an array of Tensors)
		return torch.stack([self.get_sample(label, n_features, noise_size, cond_size, k, agg_type, pool_type) for label in labels])
