class SaliencyScorer:
	'''
	This class calculates salience scores for each attribute.
	This is used to weigh the relative importance of each attribute, e.g. when generating synthetic features.
	'''
	def __init__(self, data, saliency_pow: float = 1.0, flip_saliency: bool = False):
		"""
		Calculate the saliency scores. This is a vector with the same length as the number of attributes.
		The weights are proportional to the importance (e.g. rarity) of the attributes.
		"""
		# Calculate the frequency of each attribute
		# (using flipped attributes so very frequent attributes are considered equivalent to very rare attributes)
		ratio = data.flipped_attributes.sum(0).float() / data.flipped_attributes.size(0)
		# Control
		ratio = ratio.pow(saliency_pow)
		# Normalize so that the mean is 1
		ratio /= ratio.mean()
		# Flip saliency scores if requested (so that the most common attributes, instead of the rarest, are weighted more)
		self.saliency_scores = ratio if flip_saliency else 1 - ratio
		# Print saliency scores
		print('Saliency scores:')
		for i, name in enumerate(data.attribute_names or range(len(self.saliency_scores))):
			print(name, self.saliency_scores[i])

	def apply(self, attributes):
		"""
		Apply (i.e. multiply) the saliency scores to an attributes batch.
		"""
		return attributes * self.saliency_scores.unsqueeze(0).expand(attributes.size(0), attributes.size(1)).to(attributes.device)
