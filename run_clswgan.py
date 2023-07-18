import random
import torch
import torch.backends.cudnn as cudnn
from args import parse_args
from modules.data import Data
from modules.trainer_classifier import TrainerClassifier
from modules.trainer_clswgan import TrainerClswgan
from modules.similar_sample_finder import SimilarSampleFinder

# parse arguments
args = parse_args('CLSWGAN')
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
# train a preclassifier on seen classes
train_X = data.train_X
train_Y = data.map_labels(data.train_Y, data.seen_classes)
pre_classifier = TrainerClassifier(train_X, train_Y, data, input_dim=args.n_features, batch_size=100, hidden_size=args.hidden_size,
				   n_epochs=50, n_classes=data.seen_classes.size(0), lr=0.001, beta1=0.5, is_preclassifier=True, device=device)
pre_classifier.fit_precls()
# freeze the preclassifier after training
for p in pre_classifier.model.parameters():
	p.requires_grad = False
# train the CLSWGAN
clswgan = TrainerClswgan(data, args.dataset, pre_classifier, similar_sample_finder, n_features=args.n_features, n_attributes=data.get_n_attributes(),
			 			latent_size=data.get_n_attributes(), features_per_class=args.features_per_class, batch_size=args.batch_size,
						hidden_size=args.hidden_size, n_epochs=args.n_epochs, n_classes=data.get_n_classes(),
						n_critic_iters=args.n_critic_iters, lr=args.lr, lr_cls=args.lr_cls, beta1=args.beta1,
						weight_gp=args.weight_gp, weight_precls=args.weight_precls,
						n_similar_classes=args.n_similar_classes, cond_size=args.cond_size, agg_type=args.agg_type, pool_type=args.pool_type,
						save_every=args.save_every, device=device)
clswgan.fit()
