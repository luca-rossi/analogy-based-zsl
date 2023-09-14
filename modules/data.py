import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
from sklearn import preprocessing
from typing import Tuple

DATA_PATH_TEMPLATE = '{root}/{dataset}/{file}'

class Data:
	'''
	This class loads the dataset, generates batches, and provides other useful methods.
	'''
	def __init__(self, dataset_name: str = 'awa1', dataroot: str = './data'):
		'''
		Load the dataset.
		'''
		# Get data path
		self.dataset_name = dataset_name
		self.dataroot = dataroot
		# Load features and labels
		matcontent = sio.loadmat(self.__get_data_path('res101.mat'))
		feature = matcontent['features'].T
		label = matcontent['labels'].astype(int).squeeze() - 1
		matcontent = sio.loadmat(self.__get_data_path('att_splits.mat'))
		# If there is a classes.txt file, load the class names in order
		try:
			with open(self.__get_data_path('classes.txt'), 'r') as f:
				self.class_names = [line.strip() for line in f.readlines()]
		except FileNotFoundError:
			self.class_names = None
		# If there is an attributes.txt file, load the attribute names in order
		try:
			with open(self.__get_data_path('attributes.txt'), 'r') as f:
				self.attribute_names = [line.strip() for line in f.readlines()]
		except FileNotFoundError:
			self.attribute_names = None
		# Load attributes and locations (note: numpy array index starts from 0, mat starts from 1)
		attributes = torch.from_numpy(matcontent['att'].T).float()
		train_loc = matcontent['trainval_loc'].squeeze() - 1
		test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
		test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
		# Normalize attributes (note: the datasets used here are already normalized, but we do it again just in case)
		attributes /= attributes.pow(2).sum(1).sqrt().unsqueeze(1).expand(attributes.size(0), attributes.size(1))
		self.attributes = attributes
		# Define binary attributes
		self.attribute_threshold = self.attributes.mean().to(attributes.device)
		self.binary_attributes = (attributes > self.attribute_threshold).float()
		# Define flip mask and flipped attributes (value 1 if 1s are more frequent than 0s, and 0 otherwise)
		self.flip_mask = (self.binary_attributes.sum(0) > self.binary_attributes.size(0) / 2).float()
		self.flipped_attributes = self.binary_attributes * (1 - self.flip_mask) + (1 - self.binary_attributes) * self.flip_mask
		# Transform features and labels
		scaler = preprocessing.MinMaxScaler()
		self.train_X, self.train_Y = self.__transform_features_and_labels(scaler, feature, label, train_loc, fit=True)
		self.test_seen_X, self.test_seen_Y = self.__transform_features_and_labels(scaler, feature, label, test_seen_loc)
		self.test_unseen_X, self.test_unseen_Y = self.__transform_features_and_labels(scaler, feature, label, test_unseen_loc)
		# Normalize features
		mx = self.train_X.max()
		self.train_X.mul_(1/mx)
		self.test_seen_X.mul_(1/mx)
		self.test_unseen_X.mul_(1/mx)
		# Get seen and unseen classes
		self.seen_classes = torch.from_numpy(np.unique(self.train_Y.numpy()))
		self.unseen_classes = torch.from_numpy(np.unique(self.test_unseen_Y.numpy()))
		self.all_classes = torch.cat((self.seen_classes, self.unseen_classes))
		self.dataset_size = self.train_X.size(0)
		# Print dataset info
		self.__print_dataset_info()

	def map_labels(self, label: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
		'''
		Map each element in the input label tensor to a corresponding index in the 'classes' tensor.
		The resulting 'mapped_label' tensor contains indices corresponding to the input 'classes' tensor, rather than the original class labels.
		'''
		mapped_label = torch.LongTensor(label.size())
		for i in range(classes.size(0)):
			mapped_label[label == classes[i]] = i
		return mapped_label

	def next_batch(self, batch_size: int, device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		'''
		Select a batch of data randomly from the training set.
		'''
		idx = torch.randperm(self.dataset_size)[0:batch_size]
		batch_feature = self.train_X[idx].clone()
		batch_label = self.train_Y[idx].clone()
		batch_att = self.attributes[batch_label].clone()
		batch_label = self.map_labels(batch_label, self.seen_classes)
		return batch_feature.to(device), batch_label.to(device), batch_att.to(device)

	def scale_attributes(self, scale: float) -> torch.Tensor:
		'''
		Scale attributes by the given scale factor.
		'''
		self.attributes *= scale

	def binarize_attributes(self, flip: bool = False) -> torch.Tensor:
		'''
		Binarize attributes using the given threshold. If no threshold is given, use the median value for each class.
		'''
		self.attributes = self.flipped_attributes if flip else self.binary_attributes

	def get_binary_attributes(self, attributes: torch.Tensor, flip: bool = False) -> torch.Tensor:
		'''
		Return the binary attributes for the given attributes.
		'''
		binary = (attributes > self.attribute_threshold).float()
		if flip:
			binary = binary * (1 - self.flip_mask) + (1 - binary) * self.flip_mask
		return binary

	def get_n_classes(self) -> int:
		return self.attributes.size(0)

	def get_n_attributes(self) -> int:
		return self.attributes.size(1)

	def print_class_name(self, i, mapped: bool = False):
		'''
		Print class name for the given index, if available. If 'mapped' is True, the index is mapped as if there are only seen classes.
		'''
		if self.class_names is not None:
			if mapped:
				i = self.seen_classes[i]
			print(self.class_names[i])
		else:
			print('Class name not available')

	def print_attribute_names(self, attributes: torch.Tensor, flip: bool = False):
		'''
		Given an attribute vector, print the names next to the values, if available. Also print the binary values.
		'''
		attributes = attributes.cpu()
		binary_attributes = self.get_binary_attributes(attributes, flip)
		if self.attribute_names is not None:
			for i, name in enumerate(self.attribute_names):
				flip_info = ' - Flipped' if flip and self.flip_mask[i].item() else ''
				print(f'{name}: {attributes[i]:.3f} ({binary_attributes[i].item()}){flip_info}')
		else:
			print('Attribute names not available')

	def print_attributes_info(self):
		'''
		For each attribute, print mean and variance across classes, along with the name of that attribute (if available).
		'''
		print('Attributes info:')
		for i, name in enumerate(self.attribute_names or range(self.attributes.size(1))):
			print(f'\t{name}: \tMean: {self.attributes[:, i].mean():.3f} \tVar: {self.attributes[:, i].var():.3f} \tMin: {self.attributes[:, i].min():.3f} \tMax: {self.attributes[:, i].max():.3f}')
		print('Overall mean: {:.3f}'.format(self.attributes.mean()))
		print('Overall variance: {:.3f}'.format(self.attributes.var()))

	def __get_data_path(self, file: str) -> str:
		'''
		Get the path to a data file in the dataset.
		'''
		return DATA_PATH_TEMPLATE.format(root=self.dataroot, dataset=self.dataset_name, file=file)

	def __transform_features_and_labels(self, scaler: preprocessing.MinMaxScaler, feature: torch.Tensor, label: torch.Tensor, 
									   loc: np.ndarray, fit: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
		'''
		Transform features and labels using the given scaler and location indices.
		If 'fit' is True, fit the scaler to the data before transforming.
		'''
		transformed_feature = scaler.fit_transform(feature[loc]) if fit else scaler.transform(feature[loc])
		transformed_label = label[loc]
		return torch.from_numpy(transformed_feature).float(), torch.from_numpy(transformed_label).long()

	def __print_dataset_info(self):
		'''
		Print dataset information.
		'''
		print(f'Dataset: {self.dataset_name}')
		print(f'Number of classes: {self.attributes.size(0)} (seen: {self.seen_classes.size(0)}, unseen: {self.unseen_classes.size(0)})')
		print(f'Number of attributes: {self.attributes.size(1)}')
		print(f'Training samples: {self.train_X.size(0)}')
