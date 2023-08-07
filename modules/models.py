import torch
import torch.nn as nn
from typing import Tuple

def init_weights(m: nn.Module):
	'''
	Initialize weights of the model.
	'''
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		m.weight.data.normal_(0.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class Classifier(nn.Module):
	'''
	Standard log softmax classifier.
	'''
	def __init__(self, input_dim: int, n_classes: int):
		super(Classifier, self).__init__()
		self.fc = nn.Linear(input_dim, n_classes)
		self.log = nn.LogSoftmax(dim=1)
		self.apply(init_weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.fc(x)
		h = self.log(h)
		return h

class Critic(nn.Module):
	'''
	Critic network conditioned on attributes: takes in a feature vector and an attribute vector and outputs a "realness" score.
	'''
	def __init__(self, n_features: int, n_attributes: int, hidden_size: int = 4096):
		super(Critic, self).__init__()
		self.fc1 = nn.Linear(n_features + n_attributes, hidden_size)
		self.fc2 = nn.Linear(hidden_size, 1)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.apply(init_weights)

	def forward(self, x: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
		h = torch.cat((x, att), dim=1)
		h = self.lrelu(self.fc1(h))
		h = self.fc2(h)
		return h

class Generator(nn.Module):
	'''
	Generator network conditioned on attributes: takes in a noise vector and an attribute vector and outputs a feature vector.
	Its hidden layer can optionally take in a feedback vector to improve the quality of the generated features.
	'''
	def __init__(self, n_features: int, n_attributes: int, latent_size: int, hidden_size: int = 4096, use_sigmoid: bool = False):
		super(Generator, self).__init__()
		self.fc1 = nn.Linear(n_attributes + latent_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, n_features)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.activation = nn.Sigmoid() if use_sigmoid else nn.ReLU(True)
		self.apply(init_weights)

	def forward(self, noise: torch.Tensor, att: torch.Tensor, feedback_weight: float = None, feedback: torch.Tensor = None) -> torch.Tensor:
		h = torch.cat((noise, att), dim=1)
		h = self.lrelu(self.fc1(h))
		if feedback is not None:
			h = h + feedback_weight * feedback
		h = self.activation(self.fc2(h))
		return h

class Decoder(nn.Module):
	'''
	Decoder network: takes in a feature vector and outputs an attribute vector. Two cases are possible:
	- Semantic embedding decoder network:
		- It learns to reconstruct the attribute vector using a cycle consistency reconstruction loss.
		- Its hidden representation is passed to the feedback module.
	- Feature refinement decoder network:
		- It produces a large vector h of dimension n_attributes * 2, where the first half learns to generate centroids.
		- The two halves encode means and standard deviations, and are used to reconstruct the attribute vector.
	'''
	def __init__(self, n_features: int, n_attributes: int, hidden_size: int = 4096, with_fr: bool = False):
		super(Decoder, self).__init__()
		self.with_fr = with_fr
		self.n_attributes = n_attributes
		self.fc1 = nn.Linear(n_features, hidden_size)
		if self.with_fr:
			self.fc2 = nn.Linear(hidden_size, n_attributes * 2)
			self.sigmoid = nn.Sigmoid()
		else:
			self.fc2 = nn.Linear(hidden_size, n_attributes)
		self.lrelu = nn.LeakyReLU(0.2, True)
		# Define the hidden layer to detach for the feedback module
		self.hidden_features = None
		self.apply(init_weights)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		self.hidden_features = self.lrelu(self.fc1(x))
		h = self.fc2(self.hidden_features)
		means, stds = h[:, :self.n_attributes], h[:, self.n_attributes:]
		if self.with_fr:
			stds = self.sigmoid(stds)
			h = torch.randn_like(means) * stds + means
			h = self.sigmoid(h)
		else:
			h = h / h.pow(2).sum(1).sqrt().unsqueeze(1).expand(h.size(0), h.size(1))
		return means, h

	def get_hidden_features(self) -> torch.Tensor:
		return self.hidden_features.detach()

class Encoder(nn.Module):
	'''
	VAE encoder network: takes in a feature vector and an attribute vector and outputs a distribution over the latent space.
	This distribution is used to sample the latent vector for the generator.
	'''
	def __init__(self, n_features: int, n_attributes: int, latent_size: int, hidden_size: int = 4096):
		super(Encoder, self).__init__()
		self.fc1 = nn.Linear(n_features + n_attributes, hidden_size)
		self.fc2 = nn.Linear(hidden_size, latent_size * 2)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.linear_mean = nn.Linear(latent_size * 2, latent_size)
		self.linear_log_var = nn.Linear(latent_size * 2, latent_size)
		self.apply(init_weights)

	def forward(self, x: torch.Tensor, att: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		h = torch.cat((x, att), dim=1)
		h = self.lrelu(self.fc1(h))
		h = self.lrelu(self.fc2(h))
		mean = self.linear_mean(h)
		log_var = self.linear_log_var(h)
		return mean, log_var

class Feedback(nn.Module):
	'''
	Feedback module: takes in a hidden representation from the decoder and outputs a feedback vector for the generator.
	'''
	def __init__(self, hidden_size: int = 4096):
		super(Feedback, self).__init__()
		self.fc1 = nn.Linear(hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.lrelu = nn.LeakyReLU(0.2, True)
		self.apply(init_weights)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.lrelu(self.fc1(x))
		h = self.lrelu(self.fc2(h))
		return h
