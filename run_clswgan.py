import random, operator
import torch
import torch.backends.cudnn as cudnn
from args import parse_args
from modules.data import Data
from modules.trainer_classifier import TrainerClassifier
from modules.trainer_clswgan import TrainerClswgan
from modules.similar_sample_finder import SimilarSampleFinder
from footprint.Tracker import *
tracker.project_name = "Tracker Clswgan"
tracker.output_file = "clswgan_footprint.csv"

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

tracker.start()
# load data
data = Data(dataset_name=args.dataset, dataroot=args.dataroot)


#Find salient attributes
""" result = torch.min(torch.subtract(data.attributes[1], data.attributes[2]), dim=0, keepdim=False)
print("the min is")
print(result.values)
print("the position is")
print(result.indices)
print("the original value is")
print( data.attributes[1][int(result.indices)] )
print("double check subtraction")
print(data.attributes[1][int(result.indices)] - data.attributes[2][int(result.indices)] )
 """
class Salient:
	def __init__(self, myClass, index, value):
		self.myClass = myClass
		self.index = index
		self.value = value

least_similar_attr_neg, least_similar_attr_pos= [],[]
for cur in range(data.get_n_classes()):
	objects_negative, objects_positive = [],[]
	for i in range(data.get_n_classes()):
		#most "exclusive" (distant) attribute between two classes
		result = torch.subtract(data.attributes[cur], data.attributes[i])
		result_negative = torch.min(result, dim=0, keepdim=False)
		result_positive = torch.max(result, dim=0, keepdim=False)
		objects_negative.append( Salient(i, result_negative.indices, result_negative.values) )
		objects_positive.append( Salient(i, result_positive.indices, result_positive.values) )
	least_similar_attr_neg.append(  min(objects_negative, key=operator.attrgetter('value')))
	least_similar_attr_pos.append(  max(objects_positive, key=operator.attrgetter('value')))
	#index starts from zero
	print("For the class ",(cur+1), 
       		"\n\t the most positive attribute is at index", int(least_similar_attr_pos[cur].index), 
       		"\n\t the farthest class is ",(least_similar_attr_pos[cur].myClass+1), " with an attribute gap of", least_similar_attr_pos[cur].value, 
			"\n\t the most negative attribute is at index", int(least_similar_attr_neg[cur].index), 
			"\n\t the farthest class is ",(least_similar_attr_neg[cur].myClass+1), " with an attribute gap of", least_similar_attr_neg[cur].value
		)

	
# define similar sample finder
similar_sample_finder = SimilarSampleFinder(data)
# train a preclassifier on seen classes
train_X = data.train_X
train_Y = data.map_labels(data.train_Y, data.seen_classes)
#pre_generator = GeneratorPretrainer(generator, loss_fn, optimizer)
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
tracker.stop()

#print all 312 attributes for each of 200 (CUB) classes
""" for curr in range (200):
	print("class", curr, "  ",data.attributes[curr].tolist(),"\n\n") """


#print the most similar class for each of 200 (CUB) classes
""" for cur in range(200):
	print("class ",cur, " ",similar_sample_finder.similarities[cur][0]) """


#print the attributes of the first most similar class, per each class
""" for cur in range(200):
		print("class", curr, "  ",data.attributes[similar_sample_finder.similarities[cur][0]].tolist(),"\n\n")
	for cur in range(len(data.seen_classes.tolist())):	
		print( similar_sample_finder.similarities[0][cur] )
 """
#Exaggerate the attributes from the most similar class, e.g. by multiplying them by a constant
#Put after loading data
""" for cur in range(data.get_n_classes()):
 """								#this is the most similar class to the current
""" 	torch.mul(data.attributes[int(similar_sample_finder.similarities[cur][0])], 0.5)
 """