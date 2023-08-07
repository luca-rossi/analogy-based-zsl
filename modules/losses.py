import torch
import torch.nn as nn
import torch.autograd as autograd

def loss_grad_penalty_fn(model: nn.Module, batch_real: torch.Tensor, batch_fake: torch.Tensor, 
						 batch_attributes: torch.Tensor, batch_size: int, weight_gp: float, 
						 device: torch.device = torch.device('cpu')) -> torch.Tensor:
	'''
	Compute the gradient penalty loss.
	'''
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_real.size()).to(device)
	interpolated = (alpha * batch_real + ((1 - alpha) * batch_fake)).requires_grad_(True)
	pred_interpolated = model(interpolated, batch_attributes)
	ones = torch.ones(pred_interpolated.size()).to(device)
	gradients = autograd.grad(outputs=pred_interpolated, inputs=interpolated, grad_outputs=ones,
				   create_graph=True, retain_graph=True, only_inputs=True)[0]
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * weight_gp
	return gradient_penalty

def loss_vae_fn(recon_x: torch.Tensor, x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, 
				beta: float = 1.0) -> torch.Tensor:
	'''
	Compute the VAE loss.
	'''
	bce = nn.functional.binary_cross_entropy(recon_x + 1e-12, x.detach(), reduction='sum')
	bce = bce.sum() / x.size(0)
	kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / x.size(0)
	return (bce + beta * kld)

def loss_reconstruction_fn(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
	'''
	Compute the weighted reconstruction l1 loss.
	'''
	wt = (pred - gt).pow(2)
	wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
	loss = wt * (pred - gt).abs()
	return loss.sum() / loss.size(0)

class LossMarginCenter(nn.Module):
	'''
	Compute the Self-Adaptive Margin Center loss. This loss learns a set of label centers to:
	- Minimize the distances between samples of the same class.
	- Maximize the distances between samples of different classes.
	'''
	def __init__(self, n_classes: int = 10, n_attributes: int = 312, min_margin: bool = False, 
				 device: torch.device = torch.device('cpu')):
		super(LossMarginCenter, self).__init__()
		self.n_classes = n_classes
		self.n_attributes = n_attributes
		self.min_margin = min_margin
		self.device = device
		self.centers = nn.Parameter(torch.randn(self.n_classes, self.n_attributes).to(self.device))

	def forward(self, x: torch.Tensor, labels: torch.Tensor, margin: float, 
				weight_center: float) -> torch.Tensor:
		'''
		Forward pass of the Self-Adaptive Margin Center loss.
		'''
		batch_size = x.size(0)
		
		# Compute the distance matrix between the input samples and the centers
		all_distances = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.n_classes) + \
						torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.n_classes, batch_size).t()
		all_distances.addmm_(x, self.centers.t(), beta=1, alpha=-2)
		# Define the classes and the mask
		classes = torch.arange(self.n_classes).long().to(self.device)
		mask_labels = labels.unsqueeze(1).expand(batch_size, self.n_classes)
		mask = mask_labels.eq(classes.expand(batch_size, self.n_classes))
		# Compute distances and 'other' distances for each sample
		distances = all_distances[mask]
		other_distances = None
		if not self.min_margin:
			other_distances = self.__compute_random_other_distances(all_distances, labels, classes, batch_size)
		else:
			other_distances = self.__compute_min_other_distances(all_distances, mask, batch_size)
		# Compute the loss
		loss = self.__compute_loss(margin, weight_center, distances, other_distances, batch_size)
		return loss

	def __compute_random_other_distances(self, all_distances: torch.Tensor, labels: torch.Tensor, 
										classes: torch.Tensor, batch_size: int) -> torch.Tensor:
		'''
		Compute the 'other' distances for each sample by picking a random other class for each sample.
		'''
		index = torch.randint(self.n_classes, (labels.shape[0],)).to(labels.device)
		other_labels = labels + index
		other_labels[other_labels >= self.n_classes] = other_labels[other_labels >= self.n_classes] - self.n_classes
		other_labels = other_labels.unsqueeze(1).expand(batch_size, self.n_classes)
		mask_other = other_labels.eq(classes.expand(batch_size, self.n_classes))
		return all_distances[mask_other]

	def __compute_min_other_distances(self, all_distances: torch.Tensor, mask: torch.Tensor, 
									 batch_size: int) -> torch.Tensor:
		'''
		Compute the 'other' distances for each sample by picking the minimum other distance for each sample.
		'''
		other = torch.FloatTensor(batch_size, self.n_classes - 1).cuda()
		for i in range(batch_size):
			other[i] = (all_distances[i, mask[i, :] == 0])
		return other.min(dim=1)[0]

	def __compute_loss(self, margin: float, weight_center: float, distances: torch.Tensor, 
					  other_distances: torch.Tensor, batch_size: int) -> torch.Tensor:
		'''
		Compute the Self-Adaptive Margin Center loss.
		'''
		return torch.max(margin + weight_center * distances - (1 - weight_center) * other_distances, 
						 torch.tensor(0.0).cuda()).sum() / batch_size
