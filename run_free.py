import random
import torch
import torch.backends.cudnn as cudnn
from args import parse_args
from modules.data import Data
from modules.trainer_tfvaegan import TrainerTfvaegan
from modules.similar_sample_finder import SimilarSampleFinder

# parse arguments
args = parse_args('TFVAEGAN')
# init seed and cuda
if args.seed is None:
	args.seed = random.randint(1, 10000)
print('Random Seed:', args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
	torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
# load data
data = Data(dataset_name=args.dataset, dataroot=args.dataroot)
# define similar sample finder
similar_sample_finder = SimilarSampleFinder(data)
# train the TF-VAEGAN
tfvaegan = TrainerTfvaegan(data, args.dataset, similar_sample_finder, n_features=args.n_features, n_attributes=data.get_n_attributes(),
			 			latent_size=data.get_n_attributes(), features_per_class=args.features_per_class, batch_size=args.batch_size,
						hidden_size=args.hidden_size, n_epochs=args.n_epochs, n_classes=data.get_n_classes(),
						n_critic_iters=args.n_critic_iters, n_loops=args.n_loops,
						lr=args.lr, lr_feedback=args.lr_feedback, lr_decoder=args.lr_decoder, lr_cls=args.lr_cls,
						beta1=args.beta1, freeze_dec=args.freeze_dec,
						weight_gp=args.weight_gp, weight_critic=args.weight_critic, weight_generator=args.weight_generator,
						weight_feed_train=args.weight_feed_train, weight_feed_eval=args.weight_feed_eval, weight_recons=args.weight_recons,
						n_similar_classes=args.n_similar_classes, cond_size=args.cond_size, agg_type=args.agg_type, pool_type=args.pool_type,
						save_every=args.save_every, device=device)
tfvaegan.fit()
