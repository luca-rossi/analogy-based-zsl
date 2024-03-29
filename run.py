import random
import torch
import torch.backends.cudnn as cudnn
from args import parse_args
from modules.data import Data
from modules.generator_conditioner import GeneratorConditioner
from modules.saliency_scorer import SaliencyScorer
from modules.trainer import Trainer
from modules.trainer_classifier import TrainerClassifier

def set_random_seed(seed):
	print('Random Seed:', seed)
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

def init_device():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	cudnn.benchmark = True if device.type == 'cuda' else False
	return device

def load_data(args):
	return Data(dataset_name=args.dataset, dataroot=args.dataroot)

def train_preclassifier(data, args, device):
	train_X = data.train_X
	train_Y = data.map_labels(data.train_Y, data.seen_classes)
	preclassifier = TrainerClassifier(train_X, train_Y, data, input_dim=args.n_features, batch_size=100, hidden_size=args.hidden_size,
									   n_epochs=50, n_classes=data.seen_classes.size(0), lr=0.001, beta1=0.5, is_preclassifier=True, device=device)
	preclassifier.fit_precls()
	for p in preclassifier.model.parameters():
		p.requires_grad = False
	return preclassifier

def train_model(data, args, preclassifier, conditioner, saliency_scorer, device):
	args.preclassifier = preclassifier
	args.min_margin = args.dataset == 'AWA2'
	clswgan = Trainer(data, args.dataset, conditioner, saliency_scorer, device=device, **vars(args))
	clswgan.fit()

def main():
	# Parse arguments
	args = parse_args()
	# Init seed and cuda
	set_random_seed(args.seed if args.seed is not None else random.randint(1, 10000))
	device = init_device()
	# Load data
	data = load_data(args)
	# Transform attributes
	data.scale_attributes(args.weight_attr)
	if args.binary_attr:
		data.binarize_attributes(flip=args.flip_attr)
	# Define generator conditioner
	conditioner = GeneratorConditioner(data)
	# Define saliency scorer
	saliency_scorer = SaliencyScorer(data, saliency_pow=args.saliency_pow, flip_saliency=args.flip_saliency)
	# Train a preclassifier on seen classes (if needed)
	preclassifier = train_preclassifier(data, args, device) if args.use_preclassifier else None
	# Train the model
	train_model(data, args, preclassifier, conditioner, saliency_scorer, device)

if __name__ == "__main__":
	main()
