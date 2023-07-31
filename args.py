import argparse

# Define the default values for each dataset
defaults = {
	'CLSWGAN': {
		'AWA2': {
			'seed': 9182,
			'n_classes': 50,
			'n_features': 2048,
			'n_attributes': 85,
			'latent_size': 85,
			'hidden_size': 4096,
			'features_per_class': 1800,
			'n_epochs': 30,
			'n_critic_iters': 5,
			'n_loops': 2,
			'lr': 0.00001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_precls': 0.01
		},
		'CUB': {
			'seed': 3483,
			'n_classes': 200,
			'n_features': 2048,
			'n_attributes': 312,
			'latent_size': 312,
			'hidden_size': 4096,
			'features_per_class': 300,
			'n_epochs': 56,
			'n_critic_iters': 5,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_precls': 0.01
		},
		'FLO': {
			'seed': 806,
			'n_classes': 102,
			'n_features': 2048,
			'n_attributes': 1024,
			'latent_size': 1024,
			'hidden_size': 4096,
			'features_per_class': 1200,
			'n_epochs': 80,
			'n_critic_iters': 5,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_precls': 0.1
		},
		'SUN': {
			'seed': 4115,
			'n_classes': 717,
			'n_features': 2048,
			'n_attributes': 102,
			'latent_size': 102,
			'hidden_size': 4096,
			'features_per_class': 400,
			'n_epochs': 40,
			'n_critic_iters': 5,
			'n_loops': 2,
			'lr': 0.0002,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_precls': 0.01
		}
	},
	'TFVAEGAN': {
		'AWA2': {
			'seed': 9182,
			'n_classes': 50,
			'n_features': 2048,
			'n_attributes': 85,
			'latent_size': 85,
			'hidden_size': 4096,
			'features_per_class': 1800,
			'n_epochs': 30,
			'n_critic_iters': 5,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.00001,
			'lr_feedback': 0.0001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_feed_train': 0.01,
			'weight_feed_eval': 0.01,
			'weight_recons': 0.1
		},
		'CUB': {
			'seed': 3483,
			'n_classes': 200,
			'n_features': 2048,
			'n_attributes': 312,
			'latent_size': 312,
			'hidden_size': 4096,
			'features_per_class': 300,
			'n_epochs': 56,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_feedback': 0.00001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_feed_train': 1,
			'weight_feed_eval': 1,
			'weight_recons': 0.1
		},
		'FLO': {
			'seed': 806,
			'n_classes': 102,
			'n_features': 2048,
			'n_attributes': 1024,
			'latent_size': 1024,
			'hidden_size': 4096,
			'features_per_class': 1200,
			'n_epochs': 80,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_feedback': 0.00001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_feed_train': 0.5,
			'weight_feed_eval': 0.5,
			'weight_recons': 0.01
		},
		'SUN': {
			'seed': 4115,
			'n_classes': 717,
			'n_features': 2048,
			'n_attributes': 102,
			'latent_size': 102,
			'hidden_size': 4096,
			'features_per_class': 400,
			'n_epochs': 40,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.001,
			'lr_feedback': 0.0001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.0005,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 1,
			'weight_generator': 1,
			'weight_feed_train': 0.1,
			'weight_feed_eval': 0.01,
			'weight_recons': 0.01
		}
	},
	'FREE': {
		'AWA2': {
			'seed': 9182,
			'n_classes': 50,
			'n_features': 2048,
			'n_attributes': 85,
			'latent_size': 85,
			'hidden_size': 4096,
			'features_per_class': 4600,
			'n_epochs': 30,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.00001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_precls': 0.01,
			'weight_recons': 0.001,
			'center_margin': 50,
			'weight_margin': 0.5,
			'weight_center': 0.5
		},
		'CUB': {
			'seed': 3483,
			'n_classes': 200,
			'n_features': 2048,
			'n_attributes': 312,
			'latent_size': 312,
			'hidden_size': 4096,
			'features_per_class': 700,
			'n_epochs': 56,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 64,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_precls': 0.01,
			'weight_recons': 0.001,
			'center_margin': 200,
			'weight_margin': 0.5,
			'weight_center': 0.8
		},
		'FLO': {
			'seed': 806,
			'n_classes': 102,
			'n_features': 2048,
			'n_attributes': 1024,
			'latent_size': 1024,
			'hidden_size': 4096,
			'features_per_class': 2400,
			'n_epochs': 80,
			'n_critic_iters': 5,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'batch_size': 256,
			'weight_gp': 10,
			'weight_critic': 10,
			'weight_generator': 10,
			'weight_precls': 0.01,
			'weight_recons': 0.001,
			'center_margin': 200,
			'weight_margin': 0.5,
			'weight_center': 0.8
		},
		'SUN': {
			'seed': 4115,
			'n_classes': 717,
			'n_features': 2048,
			'n_attributes': 102,
			'latent_size': 102,
			'hidden_size': 4096,
			'features_per_class': 300,
			'n_epochs': 40,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0002,
			'lr_cls': 0.0005,
			'beta1': 0.5,
			'batch_size': 512,
			'weight_gp': 10,
			'weight_critic': 1,
			'weight_generator': 1,
			'weight_precls': 0.01,
			'weight_recons': 0.1,
			'center_margin': 120,
			'weight_margin': 0.5,
			'weight_center': 0.8
		}
	}
}

# For each param, define help message
help_params = {
	'seed': 'manual seed (for reproducibility)',
	'n_classes': 'number of all classes',
	'n_features': 'size of visual features',
	'n_attributes': 'size of semantic features',
	'latent_size': 'size of the latent z vector',
	'hidden_size': 'size of the hidden layers',
	'features_per_class': 'number features to generate per class',
	'n_epochs': 'number of epochs to train for',
	'batch_size': 'input batch size',
	'n_critic_iters': 'number of critic training iterations per epoch',
	'freeze_dec': 'freeze decoder for fake samples',
	'n_loops': 'number of iterations per epoch',
	'lr': 'learning rate of the GAN',
	'lr_feedback': 'learning rate of the feedback module',
	'lr_decoder': 'learning rate of the decoder',
	'lr_cls': 'learning rate of the softmax classifier',
	'beta1': 'beta1 for adam',
	'weight_gp': 'gradient penalty regularizer',
	'weight_precls': 'preclassifier loss regularizer',
	'weight_critic': 'critic loss regularizer',
	'weight_generator': 'generator loss regularizer',
	'weight_feed_train': 'feedback output weight for training',
	'weight_feed_eval': 'feedback output weight for evaluation',
	'weight_recons': 'reconstruction loss regularizer',
	'center_margin': 'the margin in the SAMC loss',
	'weight_margin': 'the weight for the SAMC loss',
	'weight_center': 'the weight for inter-class (vs. intra-class) distance in the SAMC loss'
}

def parse_args(model='CLSWGAN'):
	# Create the first parser to get the dataset name and dataset-independent parameters
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--dataset', '-d', default='AWA2', help='dataset name (folder containing the res101.mat and att_splits.mat files)')
	parser.add_argument('--dataroot', '-p', default='../data', help='path to dataset')
	parser.add_argument('--split', '-s', default='', help='name of the split (e.g. \'_gcs\', \'_mas\', etc.)')
	parser.add_argument('--save_every', '-e', type=int, default=0, help='save the weights every n epochs (0 to disable)')
	parser.add_argument('--n_similar_classes', '-k', type=int, default=0, help='how many similar classes to use for conditional generation')
	parser.add_argument('--cond_size', '-c', type=int, default=-1, help='size of one sample in the conditioning vector, if -1 use the feature vector size')
	parser.add_argument('--agg_type', '-a', default='concat', help='how to aggregate the similar classes for conditioning (concat or mean)')
	parser.add_argument('--pool_type', '-l', default='mean', help='how to pool the similar classes for conditioning (mean, max, or first)')
	# Extra params
	parser.add_argument('--vae_beta', type=float, default=1.0, help='the beta parameter for the VAE loss')
	# Parse the arguments
	args, _ = parser.parse_known_args()
	args.dataset = 'AWA2' if args.dataset.upper() == 'AWA' else args.dataset.upper()
	# Create the second parser to get the model parameters
	subparser = argparse.ArgumentParser(add_help=True, parents=[parser])
	# Add model- and dataset-specific parameters
	for param, value in defaults[model][args.dataset].items():
		subparser.add_argument('--' + param, type=type(value), default=value, help=help_params[param])
	# Parse the new arguments
	args, _ = subparser.parse_known_args(namespace=args)
	args.dataset = 'AWA2' if args.dataset.upper() == 'AWA' else args.dataset.upper()
	# Print the arguments
	print('Arguments:')
	for param, value in vars(args).items():
		print('\t' + param + ': ' + str(value))
	return args
