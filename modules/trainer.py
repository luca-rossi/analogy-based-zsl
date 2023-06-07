class Trainer():

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

	def __load_checkpoint(self):
		'''
		Load a checkpoint if it exists.
		'''
		start_epoch = 0
		try:
			checkpoints = [f for f in os.listdir("./checkpoints" or ".ipynb_checkpoints") if f.startswith(f'CLSWGAN_{self.dataset_name}')]
			if len(checkpoints) > 0:
				print('Loading checkpoint...')
				checkpoint = torch.load(f'checkpoints/{checkpoints[0]}')
				start_epoch = checkpoint['epoch'] + 1
				self.model_generator.load_state_dict(checkpoint['model_generator'])
				self.model_critic.load_state_dict(checkpoint['model_critic'])
				self.opt_generator.load_state_dict(checkpoint['opt_generator'])
				self.opt_critic.load_state_dict(checkpoint['opt_critic'])
				self.best_gzsl_acc_seen = checkpoint['best_gzsl_acc_seen']
				self.best_gzsl_acc_unseen = checkpoint['best_gzsl_acc_unseen']
				self.best_gzsl_acc_H = checkpoint['best_gzsl_acc_H']
				self.best_zsl_acc = checkpoint['best_zsl_acc']
				torch.set_rng_state(checkpoint['random_state'])
				print('Checkpoint loaded.')
			return start_epoch
		except FileNotFoundError:
			print("No checkpoint -> skipping")
			return start_epoch

	def __save_checkpoint(self, epoch):
		'''
		Save a checkpoint.
		'''
		print('Saving checkpoint...')
		checkpoint = {
			'epoch': epoch,
			'model_generator': self.model_generator.state_dict(),
			'model_critic': self.model_critic.state_dict(),
			'opt_generator': self.opt_generator.state_dict(),
			'opt_critic': self.opt_critic.state_dict(),
			'best_gzsl_acc_seen': self.best_gzsl_acc_seen,
			'best_gzsl_acc_unseen': self.best_gzsl_acc_unseen,
			'best_gzsl_acc_H': self.best_gzsl_acc_H,
			'best_zsl_acc': self.best_zsl_acc,
			'random_state': torch.get_rng_state(),
		}
		torch.save(checkpoint, f'checkpoints/CLSWGAN_{self.dataset_name}.pt')
		print('Checkpoint saved.')