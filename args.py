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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'lr': 0.00001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'lr': 0.0002,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.00001,
			'lr_feedback': 0.0001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_feedback': 0.00001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_feedback': 0.00001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'freeze_dec': False,
			'n_loops': 2,
			'lr': 0.001,
			'lr_feedback': 0.0001,
			'lr_decoder': 0.0001,
			'lr_cls': 0.0005,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.00001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 5,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0001,
			'lr_cls': 0.001,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
			'n_epochs_cls': 25,
			'n_critic_iters': 1,
			'freeze_dec': True,
			'n_loops': 2,
			'lr': 0.0002,
			'lr_cls': 0.0005,
			'beta1': 0.5,
			'beta1_cls': 0.5,
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
	'n_epochs': 'number of epochs for the GAN',
	'n_epochs_cls': 'number of epochs for the softmax classifier',
	'batch_size': 'input batch size',
	'n_critic_iters': 'number of critic training iterations per epoch',
	'freeze_dec': 'freeze decoder for fake samples',
	'n_loops': 'number of iterations per epoch',
	'lr': 'learning rate of the GAN',
	'lr_feedback': 'learning rate of the feedback module',
	'lr_decoder': 'learning rate of the decoder',
	'lr_cls': 'learning rate of the softmax classifier',
	'beta1': 'beta1 for the Adam optimizers of the GAN',
	'beta1_cls': 'beta1 for the Adam optimizer of the softmax classifier',
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

def clean_dataset_name(dataset_name):
	return 'AWA2' if dataset_name.upper() == 'AWA' else dataset_name.upper()

def parse_args():
	# Create the first parser to get the dataset name and dataset-independent parameters
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--model', '-m', default='', help='sets default model parameters (C = CLSWGAN, T = TFVAEGAN, F = FREE)')
	parser.add_argument('--dataset', '-d', default='AWA2', help='dataset name (folder containing the res101.mat and att_splits.mat files)')
	parser.add_argument('--dataroot', '-r', default='./data', help='path to dataset')
	parser.add_argument('--save_every', '-e', type=int, default=0, help='save the weights every n epochs (0 to disable)')
	parser.add_argument('--n_similar_classes', '-k', type=int, default=0, help='how many similar classes to use for conditional generation')
	parser.add_argument('--cond_size', '-c', type=int, default=-1, help='size of one sample in the conditioning vector, if -1 use the feature vector size')
	parser.add_argument('--noise_size', '-z', type=int, default=-1, help='size of the noise vector, if -1 use the attribute vector size')
	parser.add_argument('--agg_type', '-a', default='concat', help='how to aggregate the similar classes for conditioning (concat or mean)')
	parser.add_argument('--pool_type', '-l', default='mean', help='how to pool the similar classes for conditioning (mean, max, or first)')
	parser.add_argument('--use_preclassifier', '-p', action='store_true', help='use a preclassifier like in CLSWGAN')
	parser.add_argument('--use_feedback', '-f', action='store_true', help='use a feedback module like in TFVAEGAN')
	parser.add_argument('--use_encoder', '-n', action='store_true', help='use an encoder like in TFVAEGAN and FREE')
	parser.add_argument('--use_margin', '-g', action='store_true', help='use the SAMC loss like in FREE')
	# Extra params
	parser.add_argument('--synthesize_all', '-y', action='store_true', help='if True, synthesize all classes, otherwise only the unseen classes')
	parser.add_argument('--vae_beta', type=float, default=1.0, help='the beta parameter for the VAE loss')
	parser.add_argument('--adjust_weight_gp', action='store_true', help='dynamically adjust the weight of the gradient penalty')
	parser.add_argument('--weight_reg_recons', type=float, default=0.0, help='reconstruction loss regularizer')
	parser.add_argument('--weight_reg_generator', type=float, default=0.0, help='generator L2 regularizer')
	parser.add_argument('--binary_attr', action='store_true', help='binarize the attributes')
	parser.add_argument('--flip_attr', action='store_true', help='if binarized attributes have more 1s than 0s, flip them')
	parser.add_argument('--weight_attr', type=float, default=1.0, help='how much to scale the attribute vector')
	parser.add_argument('--use_saliency', action='store_true', help='use saliency scores to weight the attributes')
	# Parse the arguments
	args, _ = parser.parse_known_args()
	# Clean values
	args.model = args.model.lower()
	args.model = 'CLSWGAN' if args.model == 'c' else 'TFVAEGAN' if args.model == 't' else 'FREE' if args.model == 'f' else args.model
	args.dataset = clean_dataset_name(args.dataset)
	args.agg_type = args.agg_type.lower()
	args.pool_type = args.pool_type.lower()
	# If the model is specified, override parameters
	if args.model == 'CLSWGAN':
		args.use_preclassifier = True
		args.use_feedback = False
		args.use_encoder = False
		args.use_margin = False
	elif args.model == 'TFVAEGAN':
		args.use_preclassifier = False
		args.use_feedback = True
		args.use_encoder = True
		args.use_margin = False
	elif args.model == 'FREE':
		args.use_preclassifier = False
		args.use_feedback = False
		args.use_encoder = True
		args.use_margin = True
	# Create the second parser to get the model parameters
	subparser = argparse.ArgumentParser(add_help=True, parents=[parser])
	# If args.model is among the keys of the defaults dictionary, use those model- and dataset-specific parameters
	if args.model in defaults.keys():
		for param, value in defaults[args.model][args.dataset].items():
			subparser.add_argument('--' + param, type=type(value), default=value, help=help_params[param])
	# Parse the new arguments
	args, _ = subparser.parse_known_args(namespace=args)
	# Clean values
	args.dataset = clean_dataset_name(args.dataset)
	if args.cond_size == -1:
		args.cond_size = args.n_features
	if args.noise_size == -1:
		args.noise_size = args.n_attributes
	# Print the arguments
	print('Arguments:')
	for param, value in vars(args).items():
		print('\t' + param + ': ' + str(value))
	return args
