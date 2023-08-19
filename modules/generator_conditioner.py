import numpy as np
import random
import torch
from typing import List, Tuple

class GeneratorConditioner:
	'''
	This class builds the vector used to condition the generator.
	It finds the most similar class to the current one. There are two parts to this:
	- Find the most similar seen class to the current one. This is done only once at the beginning.
	- Sample a feature vector from the most similar seen class. This is done whenever we need to condition the GAN. This needs to be efficient.
	'''
	def __init__(self, data):
		'''
		Here, we find the most similar seen classes for each class.
		We simply define a matrix of labels, where each row corresponds to a class and each column corresponds to the index of a seen class sorted by similarity.
		The similarity is measured as the distance between the signatures (no features are needed, only attributes).
		'''
		# Get data and number of classes
		self.data = data
		self.n_classes = data.get_n_classes()
		# Compute similarities between classes based on their signatures
		self.similarities = np.zeros((self.n_classes, len(self.data.seen_classes)))
		for i in range(self.n_classes):
			norms, indexes = self.__compute_norms_and_indexes(i)
			# Sort the norms and indexes in ascending order
			sorted_norms, sorted_indexes = zip(*sorted(zip(norms, indexes)))
			# Store the sorted indexes in the similarities matrix
			self.similarities[i] = sorted_indexes

	def get_vector(self, labels: List[int], attributes: torch.Tensor, n_features: int, noise_size: int, cond_size: int,
				   noise: torch.Tensor = None, k: int = 0, agg_type: str = 'concat', pool_type: str = 'mean') -> torch.Tensor:
		'''
		Given a batch of labels, returns a batch of features from the most similar seen classes.
		'''
		conditioning_vector = torch.Tensor().to(attributes.device)
		# Stack features from the most similar seen classes
		if k > 0:
			conditioning_vector = torch.stack([self.__get_sample(label, n_features, cond_size, k, agg_type, pool_type) for label in labels]).to(attributes.device)
		# Concatenate attributes to the conditioning vector
		conditioning_vector = torch.cat((conditioning_vector, attributes), dim=1)
		# Concatenate noise to the conditioning vector
		if noise is None:
			noise = torch.normal(0, 1, (len(labels), noise_size)).to(attributes.device)
		conditioning_vector = torch.cat((conditioning_vector, noise), dim=1)
		return conditioning_vector

	def __compute_norms_and_indexes(self, i: int) -> Tuple[List[float], List[int]]:
		'''
		Compute norms and indexes for a given class. The distance between two classes is the norm of the difference between their signatures.
		'''
		# Get the class signature
		signature = self.data.attributes[i]
		# Compare this signature to all the signatures of the seen classes (excluding the current one)
		norms = []
		indexes = []
		for j in self.data.seen_classes:
			# If the class is the current one, use infinity norm to make sure it is not selected
			comparison_norm = np.inf if j == i else np.linalg.norm(signature - self.data.attributes[j])
			indexes.append(j)
			norms.append(comparison_norm)
		return norms, indexes

	def __get_sample(self, label: int, n_features: int, cond_size: int,
		  			 k: int = 0, agg_type: str = 'concat', pool_type: str = 'mean') -> torch.Tensor:
		'''
		Given a label, a number of similar classes to use (default 1), and the type of aggregation (concat or mean) and pooling (mean, max, or first),
		returns a feature vector from the most similar seen class or a fused feature vector from the most similar seen classes.
		'''
		# Let's make the length of every feature vector equal cond_size
		pooling_size = n_features // cond_size
		# Get the k most similar seen classes
		similar_labels = self.similarities[label][:k]
		# Generate the conditioning feature vectors from the most similar seen classes
		feature_vectors = [self.__get_feature_vector(similar_label, pooling_size, pool_type) for similar_label in similar_labels]
		# Aggregate the feature vectors into a single feature vector
		feature_vector = self.__aggregate_feature_vectors(feature_vectors, agg_type)
		return feature_vector

	def __get_feature_vector(self, similar_label: int, pooling_size: int, pool_type: str) -> torch.Tensor:
		'''
		Get a feature vector for a given similar label.
		'''
		# Get a random location from the occurrences of the similar label
		location = random.choice(np.where(self.data.train_Y == similar_label)[0])
		# Get the feature vector at that location and apply pooling to reduce dimensionality
		feature_vector = self.data.train_X[location].view(-1, pooling_size)
		if pool_type == 'mean':
			feature_vector = torch.mean(feature_vector, dim=1)
		elif pool_type == 'max':
			feature_vector = torch.max(feature_vector, dim=1).values
		elif pool_type == 'first':
			feature_vector = feature_vector[:, 0]
		else:
			raise ValueError('Invalid pooling type')
		return feature_vector

	def __aggregate_feature_vectors(self, feature_vectors: List[torch.Tensor], agg_type: str) -> torch.Tensor:
		'''
		Aggregate feature vectors by concatenating or averaging.
		'''
		if agg_type == 'concat':
			feature_vector = torch.stack(feature_vectors, dim=0).flatten()
		elif agg_type == 'mean':
			feature_vector = torch.mean(torch.stack(feature_vectors, dim=0), dim=0)
		else:
			raise ValueError('Invalid aggregation type.')
		return feature_vector
