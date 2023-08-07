import os
import torch
from modules.losses import loss_grad_penalty_fn, loss_vae_fn, loss_reconstruction_fn, LossMarginCenter
from modules.models import Generator, Critic, Encoder, Decoder, Feedback
from modules.trainer_classifier import TrainerClassifier
from typing import Tuple

class Trainer():
	'''
	This class implements the training and evaluation of the model.
	'''
	def __init__(self, data, dataset_name, similar_sample_finder, device: torch.device = torch.device('cpu'), **kwargs):
		'''
		Setup models, optimizers, and other parameters.
		'''
		# Init model parameters
		self.use_preclassifier = kwargs.get('use_preclassifier', False)
		self.use_feedback = kwargs.get('use_feedback', False)
		self.use_encoder = kwargs.get('use_encoder', False)
		self.use_margin = kwargs.get('use_margin', False)
		# Init basic parameters
		self.data = data
		self.dataset_name = dataset_name
		self.n_features = kwargs.get('n_features', 2048)
		self.n_attributes = kwargs.get('n_attributes', 85)
		self.features_per_class = kwargs.get('features_per_class', 1800)
		self.batch_size = kwargs.get('batch_size', 64)
		self.hidden_size = kwargs.get('hidden_size', 4096)
		self.n_epochs = kwargs.get('n_epochs', 30)
		self.n_classes = kwargs.get('n_classes', 50)
		self.n_loops = kwargs.get('n_loops', 1)
		self.n_critic_iters = kwargs.get('n_critic_iters', 5)
		self.weight_critic = kwargs.get('weight_critic', 1)
		self.weight_generator = kwargs.get('weight_generator', 1)
		self.lr = kwargs.get('lr', 0.001)
		self.lr_cls = kwargs.get('lr_cls', 0.001)
		self.lr_feedback = kwargs.get('lr_feedback', 0.0001)
		self.lr_decoder = kwargs.get('lr_decoder', 0.0001)
		self.beta1 = kwargs.get('beta1', 0.5)
		self.weight_gp = kwargs.get('weight_gp', 10)
		self.adjust_weight_gp = kwargs.get('adjust_weight_gp', False)
		self.save_every = kwargs.get('save_every', 0)
		self.device = device
		self.verbose = kwargs.get('verbose', False)
		# Init model-specific parameters...
		# ... preclassifier
		self.weight_precls = kwargs.get('weight_precls', 1)
		self.preclassifier = kwargs.get('preclassifier', None)
		# ... feedback
		self.weight_feed_train = kwargs.get('weight_feed_train', 0.1)
		self.weight_feed_eval = kwargs.get('weight_feed_eval', 0.1)
		# ... encoder
		self.freeze_dec = kwargs.get('freeze_dec', False)
		self.weight_gp = kwargs.get('weight_gp', 10)
		self.weight_recons = kwargs.get('weight_recons', 1)
		self.vae_beta = kwargs.get('vae_beta', 1)
		# ... margin
		self.center_margin = kwargs.get('center_margin', 0.2)
		self.weight_margin = kwargs.get('weight_margin', 0.5)
		self.weight_center = kwargs.get('weight_center', 0.5)
		self.min_margin = kwargs.get('min_margin', False)
		# Init generator conditioning parameters
		self.similar_sample_finder = similar_sample_finder
		self.n_similar_classes = kwargs.get('n_similar_classes', 0)
		self.agg_type = kwargs.get('agg_type', 'concat')
		self.pool_type = kwargs.get('pool_type', 'mean')
		self.noise_size = kwargs.get('noise_size', self.n_attributes)
		self.cond_size = kwargs.get('cond_size', self.n_features)
		self.latent_size = self.noise_size + (self.cond_size * self.n_similar_classes if self.agg_type == 'concat' else self.cond_size)
		# Init models, losses, and optimizers
		self.model_generator = Generator(self.n_features, self.n_attributes, self.latent_size, self.hidden_size, use_sigmoid=self.use_encoder).to(self.device)
		self.opt_generator = torch.optim.Adam(self.model_generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
		print(self.model_generator)
		self.model_critic = Critic(self.n_features, self.n_attributes, self.hidden_size).to(self.device)
		self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
		print(self.model_critic)
		if self.use_preclassifier:
			self.loss_classifier_fn = torch.nn.NLLLoss().to(self.device)
		if self.use_encoder:
			self.model_encoder = Encoder(self.n_features, self.n_attributes, self.latent_size, self.hidden_size).to(self.device)
			self.opt_encoder = torch.optim.Adam(self.model_encoder.parameters(), lr=self.lr)
			print(self.model_encoder)
			self.model_decoder = Decoder(self.n_features, self.n_attributes, self.hidden_size, with_fr=self.use_margin).to(self.device)
			self.opt_decoder = torch.optim.Adam(self.model_decoder.parameters(), lr=self.lr_decoder, betas=(self.beta1, 0.999))
			print(self.model_decoder)
		if self.use_feedback:
			self.model_feedback = Feedback(self.hidden_size).to(self.device)
			self.opt_feedback = torch.optim.Adam(self.model_feedback.parameters(), lr=self.lr_feedback, betas=(self.beta1, 0.999))
			print(self.model_feedback)
		if self.use_margin:
			self.center_criterion = LossMarginCenter(n_classes=self.data.seen_classes.size(0), n_attributes=self.n_attributes, min_margin=self.min_margin, device=self.device)
			self.opt_center = torch.optim.Adam(self.center_criterion.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
		# Init tensors
		self.batch_features = torch.FloatTensor(self.batch_size, self.n_features).to(self.device)
		self.batch_attributes = torch.FloatTensor(self.batch_size, self.n_attributes).to(self.device)
		self.batch_labels = torch.LongTensor(self.batch_size).to(self.device)
		self.one = torch.tensor(1, dtype=torch.float).to(self.device)
		self.mone = self.one * -1

	def fit(self):
		'''
		Train the model. Both ZSL and GZSL performance are evaluated at each epoch.
		'''
		self.best_gzsl_acc_seen = 0
		self.best_gzsl_acc_unseen = 0
		self.best_gzsl_acc_H = 0
		self.best_zsl_acc = 0
		start_epoch = self.__load_checkpoint()
		for epoch in range(start_epoch, self.n_epochs):
			self.__train_epoch(epoch)
			self.__eval_epoch()
			if self.save_every > 0 and epoch > 0 and (epoch + 1) % self.save_every == 0:
				self.__save_checkpoint(epoch)
		print('Dataset', self.dataset_name)
		print('The best ZSL unseen accuracy is %.4f' % self.best_zsl_acc.item())
		print('The best GZSL seen accuracy is %.4f' % self.best_gzsl_acc_seen.item())
		print('The best GZSL unseen accuracy is %.4f' % self.best_gzsl_acc_unseen.item())
		print('The best GZSL H is %.4f' % self.best_gzsl_acc_H.item())

	def __load_checkpoint(self) -> int:
		'''
		Load a checkpoint if it exists.
		'''
		start_epoch = 0
		try:
			checkpoints = [f for f in os.listdir('checkpoints') if f.startswith(f'checkpoint_{self.dataset_name}')]
			if len(checkpoints) > 0:
				print('Loading checkpoint...')
				checkpoint = torch.load(f'checkpoints/{checkpoints[0]}')
				start_epoch = checkpoint['epoch'] + 1
				self.model_generator.load_state_dict(checkpoint['model_generator'])
				self.opt_generator.load_state_dict(checkpoint['opt_generator'])
				self.model_critic.load_state_dict(checkpoint['model_critic'])
				self.opt_critic.load_state_dict(checkpoint['opt_critic'])
				if self.use_encoder:
					self.model_encoder.load_state_dict(checkpoint['model_encoder'])
					self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
					self.model_decoder.load_state_dict(checkpoint['model_decoder'])
					self.opt_decoder.load_state_dict(checkpoint['opt_decoder'])
				if self.use_feedback:
					self.model_feedback.load_state_dict(checkpoint['model_feedback'])
					self.opt_feedback.load_state_dict(checkpoint['opt_feedback'])
				if self.use_margin:
					self.center_criterion.load_state_dict(checkpoint['model_center'])
					self.opt_center.load_state_dict(checkpoint['opt_center'])
				self.best_gzsl_acc_seen = checkpoint['best_gzsl_acc_seen']
				self.best_gzsl_acc_unseen = checkpoint['best_gzsl_acc_unseen']
				self.best_gzsl_acc_H = checkpoint['best_gzsl_acc_H']
				self.best_zsl_acc = checkpoint['best_zsl_acc']
				torch.set_rng_state(checkpoint['random_state'])
				print('Checkpoint loaded.')
		except FileNotFoundError:
			print("No checkpoint to load.")
		return start_epoch

	def __save_checkpoint(self, epoch: int):
		'''
		Save a checkpoint.
		'''
		print('Saving checkpoint...')
		checkpoint = dict()
		checkpoint['epoch'] = epoch
		checkpoint['model_generator'] = self.model_generator.state_dict()
		checkpoint['opt_generator'] = self.opt_generator.state_dict()
		checkpoint['model_critic'] = self.model_critic.state_dict()
		checkpoint['opt_critic'] = self.opt_critic.state_dict()
		if self.use_encoder:
			checkpoint['model_encoder'] = self.model_encoder.state_dict()
			checkpoint['opt_encoder'] = self.opt_encoder.state_dict()
			checkpoint['model_decoder'] = self.model_decoder.state_dict()
			checkpoint['opt_decoder'] = self.opt_decoder.state_dict()
		if self.use_feedback:
			checkpoint['model_feedback'] = self.model_feedback.state_dict()
			checkpoint['opt_feedback'] = self.opt_feedback.state_dict()
		if self.use_margin:
			checkpoint['model_center'] = self.center_criterion.state_dict()
			checkpoint['opt_center'] = self.opt_center.state_dict()
		checkpoint['best_gzsl_acc_seen'] = self.best_gzsl_acc_seen
		checkpoint['best_gzsl_acc_unseen'] = self.best_gzsl_acc_unseen
		checkpoint['best_gzsl_acc_H'] = self.best_gzsl_acc_H
		checkpoint['best_zsl_acc'] = self.best_zsl_acc
		checkpoint['random_state'] = torch.get_rng_state()
		torch.save(checkpoint, f'checkpoints/checkpoint_{self.dataset_name}.pt')
		print('Checkpoint saved.')

	def __freeze(self, *models):
		'''
		Freeze the parameters of the given models.
		'''
		for model in models:
			for p in model.parameters():
				p.requires_grad = False

	def __unfreeze(self, *models):
		'''
		Unfreeze the parameters of the given models.
		'''
		for model in models:
			for p in model.parameters():
				p.requires_grad = True

	def __train_epoch(self, epoch: int):
		'''
		Train the models for one epoch.
		'''
		losses = {}
		for n_loop in range(0, self.n_loops):
			for i in range(0, self.data.dataset_size, self.batch_size):
				# Unfreeze the critic and decoder parameters for training
				self.__unfreeze(self.model_critic)
				if self.use_encoder:
					self.__unfreeze(self.model_decoder)
				# Train the decoder and critic for n_critic_iters steps
				gradient_penalties = []
				for _ in range(self.n_critic_iters):
					# Sample a mini-batch
					self.batch_features, self.batch_labels, self.batch_attributes = self.data.next_batch(self.batch_size, device=self.device)
					# Train decoder
					if self.use_encoder:
						_ = self.__decoder_step()
					# Train critic
					critic_losses, gp = self.__critic_step(n_loop)
					losses.update(critic_losses)
					gradient_penalties.append(gp)
				# Dynamically adjust the weight of the gradient penalty
				if self.adjust_weight_gp:
					self.__adjust_weight_gp(gradient_penalties)
				# Freeze the critic and decoder parameters to train the generator
				self.__freeze(self.model_critic)
				if self.use_encoder and self.weight_recons > 0 and self.freeze_dec:
					self.__freeze(self.model_decoder)
				# Train the generator, the encoder, the feedback module, and the decoder again
				generator_losses = self.__generator_step(n_loop)
				losses.update(generator_losses)
				# Show progress
				if self.verbose and i % (self.batch_size * 5) == 0:
					if self.n_loops == 1:
						print('%d/%d' % (i, self.data.dataset_size))
					else:
						print('%d/%d - %d/%d' % (n_loop + 1, self.n_loops, i, self.data.dataset_size))
		# Print epoch info
		print('[%d/%d]' % (epoch + 1, self.n_epochs), end=' ')
		loss_info = []
		print('Losses - ', end='')
		for k, v in losses.items():
			loss_info.append('%s: %.4f,' % (k, v.data.item()))
		print(' '.join(loss_info)[:-1])

	def __decoder_step(self) -> dict:
		'''
		Train the decoder for one step to reconstruct the attributes of the real batch with a reconstruction loss.
		'''
		# Train with real batch
		self.model_decoder.zero_grad()
		means, recons = self.model_decoder(self.batch_features)
		loss_decoder = self.weight_recons * loss_reconstruction_fn(recons, self.batch_attributes)
		# SAMC loss through means
		if self.use_margin:
			center_loss_real = self.center_criterion(means, self.batch_labels, margin=self.center_margin, weight_center=self.weight_center)
			loss_decoder += center_loss_real * self.weight_margin
			loss_decoder.backward()
			self.opt_center.step()
		else:
			loss_decoder.backward()
		self.opt_decoder.step()
		losses = {'decoder': loss_decoder}
		return losses

	def __critic_step(self, n_loop: int = 0) -> Tuple[dict, float]:
		'''
		Train the critic for one step on two mini-batches: a real one from the dataset, and a synthetic one from the generator.
		'''
		# Train with real batch
		self.model_critic.zero_grad()
		critic_real = self.model_critic(self.batch_features, self.batch_attributes)
		critic_real = self.weight_critic * critic_real.mean()
		critic_real.backward(self.mone)
		# Train with fake batch
		fake, _, _ = self.__generate_from_features(n_loop)
		critic_fake = self.model_critic(fake.detach(), self.batch_attributes)
		critic_fake = self.weight_critic * critic_fake.mean()
		critic_fake.backward(self.one)
		# Gradient penalty
		gradient_penalty = self.weight_critic * loss_grad_penalty_fn(self.model_critic, self.batch_features, fake.data, self.batch_attributes, self.batch_size, self.weight_gp, self.device)
		gradient_penalty.backward()
		# Loss
		wasserstein = critic_real - critic_fake
		loss_critic = -wasserstein + gradient_penalty
		self.opt_critic.step()
		losses = {'critic': loss_critic, 'Wasserstein': wasserstein}
		return losses, gradient_penalty.data

	def __generator_step(self, n_loop: int = 0) -> dict:
		'''
		Train the generator for one step. Include the classification loss from the pre-classifier.
		The generator is trained to fool the critic, and the encoder learns a distribution over a latent space for the generator.
		Optionally, train the encoder, the feedback module, and the decoder as well for one step.
		'''
		losses = {}
		self.model_generator.zero_grad()
		if self.use_feedback:
			self.model_feedback.zero_grad()
		if self.use_encoder:
			self.model_encoder.zero_grad()
		# Generate a fake batch from a latent distribution, learned from real features
		fake, means, log_var = self.__generate_from_features(n_loop)
		if self.use_encoder:
			# VAE loss
			loss_vae = loss_vae_fn(fake, self.batch_features, means, log_var, beta=self.vae_beta)
			losses['VAE'] = loss_vae
		# Generator loss from the critic's evaluation
		critic_fake = self.model_critic(fake, self.batch_attributes).mean()
		loss_generator = -critic_fake
		loss_generator_tot = self.weight_generator * loss_generator
		# Decoder and reconstruction loss
		if self.use_encoder:
			self.model_decoder.zero_grad()
			_, recons_fake = self.model_decoder(fake)
			loss_recons = loss_reconstruction_fn(recons_fake, self.batch_attributes)
			loss_generator_tot += loss_vae
			loss_generator_tot += self.weight_recons * loss_recons
		# Preclassifier loss
		if self.use_preclassifier:
			loss_classifier = self.loss_classifier_fn(self.preclassifier.model(fake), self.batch_labels)
			losses['preclassifier'] = loss_classifier
			loss_generator_tot += self.weight_precls * loss_classifier
		# Total loss
		loss_generator_tot.backward()
		self.opt_generator.step()
		if self.use_encoder:
			self.opt_encoder.step()
			if self.use_feedback and n_loop >= 1:
				self.opt_feedback.step()
			if self.weight_recons > 0 and not self.freeze_dec:
				self.opt_decoder.step()
		losses['generator'] = loss_generator
		return losses

	def __generate_from_features(self, n_loop: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		'''
		Use the generator to synthesize a batch from a latent distribution, which is learned from real features by the decoder.
		Improve the generated features with the feedback module from the second feedback loop onward.
		'''
		means, log_var = None, None
		if self.use_encoder:
			# Use real features to generate a latent distribution with the encoder
			means, log_var = self.model_encoder(self.batch_features, self.batch_attributes)
		# Conditional input for the generator
		gen_input = self.similar_sample_finder.get_samples(self.batch_labels, self.n_features, self.noise_size, self.cond_size, k=self.n_similar_classes, agg_type=self.agg_type, pool_type=self.pool_type).to(self.device)
		# Generate a fake batch with the generator from the latent distribution
		fake = self.model_generator(gen_input, self.batch_attributes)
		# From the second feedback loop onward, improve the generated features with the feedback module
		if self.use_feedback and n_loop >= 1:
			# Call the forward function of decoder to get the hidden features
			_ = self.model_decoder(fake)
			decoder_features = self.model_decoder.get_hidden_features()
			feedback = self.model_feedback(decoder_features)
			fake = self.model_generator(gen_input, self.batch_attributes, feedback_weight=self.weight_feed_train, feedback=feedback)
		return fake, means, log_var

	def __adjust_weight_gp(self, gradient_penalties):
		'''
		Dynamically adjust the weight of the gradient penalty to keep it in a reasonable range.
		'''
		# Average the gradient penalties
		gp = sum(gradient_penalties) / (self.weight_critic * self.weight_gp * self.n_critic_iters)
		if (gp > 1.05).sum() > 0:
			self.weight_gp *= 1.1
			# print('GP weight increased to %.4f' % self.weight_gp)
		elif (gp < 1.001).sum() > 0:
			self.weight_gp /= 1.1
			# print('GP weight decreased to %.4f' % self.weight_gp)

	def __eval_epoch(self):
		'''
		Evaluate the model at the end of each epoch, both ZSL and GZSL.
		The generator is used to generate unseen features, then a classifier is trained and evaluated on the (partially) synthetic dataset.
		'''
		# Evaluation mode
		self.model_generator.eval()
		feedback = None
		decoder = None
		is_decoder_fr = self.use_encoder
		if self.use_feedback:
			self.model_feedback.eval()
			feedback = self.model_feedback
		if self.use_encoder:
			self.model_decoder.eval()
			decoder = self.model_decoder
		# Generate synthetic features
		syn_X, syn_Y = self.data.generate_syn_features(self.model_generator, self.data.unseen_classes, self.data.attributes,
								self.features_per_class, self.n_features, self.n_attributes, self.latent_size, self.device,
								model_decoder=decoder, model_feedback=feedback, feedback_weight=self.weight_feed_eval)
		# GZSL evaluation: concatenate real seen features with synthesized unseen features, then train and evaluate a classifier
		train_X = torch.cat((self.data.train_X, syn_X), 0)
		train_Y = torch.cat((self.data.train_Y, syn_Y), 0)
		cls = TrainerClassifier(train_X, train_Y, self.data, n_attributes=self.n_attributes, batch_size=self.features_per_class,
				hidden_size=self.hidden_size, n_epochs=25, n_classes=self.n_classes, lr=self.lr_cls, beta1=0.5,
				model_decoder=decoder, is_decoder_fr=is_decoder_fr, device=self.device)
		acc_seen, acc_unseen, acc_H = cls.fit_gzsl()
		if self.best_gzsl_acc_H < acc_H:
			self.best_gzsl_acc_seen, self.best_gzsl_acc_unseen, self.best_gzsl_acc_H = acc_seen, acc_unseen, acc_H
		print('GZSL - Seen: %.4f, Unseen: %.4f, H: %.4f' % (acc_seen, acc_unseen, acc_H))
		# ZSL evaluation: use only synthesized unseen features, then train and evaluate a classifier
		train_X = syn_X
		train_Y = self.data.map_labels(syn_Y, self.data.unseen_classes)
		cls = TrainerClassifier(train_X, train_Y, self.data, n_attributes=self.n_attributes, batch_size=self.features_per_class,
				hidden_size=self.hidden_size, n_epochs=25, n_classes=self.data.unseen_classes.size(0), lr=self.lr_cls, beta1=0.5,
				model_decoder=decoder, is_decoder_fr=is_decoder_fr, device=self.device)
		acc = cls.fit_zsl()
		if self.best_zsl_acc < acc:
			self.best_zsl_acc = acc
		print('ZSL - Unseen: %.4f' % (acc))
		# Training mode
		self.model_generator.train()
		if self.use_feedback:
			self.model_feedback.train()
		if self.use_encoder:
			self.model_decoder.train()
