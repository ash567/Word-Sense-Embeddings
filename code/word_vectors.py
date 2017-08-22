import gensim, logging
from gensim.models import Word2Vec
from random import shuffle
import numpy


f = open('wnet30_dict.txt', 'r')
d = {}
for l in f:
	ls = l.split()
	synsets = []
	for s in ls[1:]:
		synset, count = s.split(':')
		synsets.append((synset, int(count)))
	d[ls[0]] = synsets


# shows the progress during the training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
training_sentences = open("artificial_corpus.txt")
training_sentences.readline()
input_sentences = [line.lower().split() for line in training_sentences]
# TRAINING
size = 100
window = 3
min_count = 5
workers = 3
alpha = 0.04
sample = 10.0**(-3)
seed = 6
sg = 1
hs = 1
negative = 5

iter_no = 5

model =	Word2Vec(iter = 1,
			size = size,
			window = window,
			min_count = min_count,
			workers = workers,
			alpha = alpha,
			min_alpha = alpha,
			seed = seed,
			sg = sg,
			hs = hs,
			negative = negative,
			# sample = sample
			)

model.build_vocab(input_sentences)

# for epoch in range(iter_no):
# 	print '\n\n'
# 	print "Epoch number:", epoch, '\n\n'
# 	shuffle(input_sentences)
# 	model.train(input_sentences)
# 	model.alpha = model.alpha / (1 + epoch)
# 	model.min_alpha = model.alpha

# model.save('random_walk_model')

model = Word2Vec.load('random_walk_model')

def similarity(w1, w2, add_n = 0):
	if w1 not in d or w2 not in d:
		return None
	sum_w1 = sum([add_n+count for synset, count in d[w1]])
	sum_w2 = sum([add_n+count for synset, count in d[w2]])
	sim_sum = 0
	for syn1, count1 in d[w1]:
		for syn2, count2 in d[w2]:
			sim_sum += model.similarity(syn1, syn2)*(float(count1+add_n)/sum_w1)*(float(count2+add_n)/sum_w2)
	return sim_sum

# Much better
def similarity2(w1, w2):
	if w1 not in d or w2 not in d:
		return None
	best_sim = 0
	for syn1, _ in d[w1]:
		for syn2, _ in d[w2]:
			best_sim = max(best_sim, model.similarity(syn1, syn2))
	return best_sim

# Useless
def similarity3(w1, w2, add_n = 1, alpha = 1):
	if w1 not in d or w2 not in d:
		return None
	sum_w1 = sum([add_n+count for synset, count in d[w1]])
	sum_w2 = sum([add_n+count for synset, count in d[w2]])
	sim_sum = 0
	for syn1, count1 in d[w1]:
		v1 = model[syn1]
		for syn2, count2 in d[w2]:
			v2 = model[syn2]
			T = numpy.dot(v1, v2)/(numpy.dot(v1, v1) + numpy.dot(v2, v2) - numpy.dot(v1, v2))
			sim_sum += (float(count1+add_n)/sum_w1)*(float(count2+add_n)/sum_w2)*(T**alpha)
	return sim_sum

def cosine(v1, v2):
	return numpy.dot(v1, v2)/(numpy.dot(v1, v1) ** 0.5)/(numpy.dot(v2, v2) ** 0.5)

def relational_similarity(w1, w2, w3, w4, add_n = 0):
	best_sim = 0
	v1, v2 = None, None
	for syn1, _ in d[w1]:
		for syn2, _ in d[w2]:
			if model.similarity(syn1, syn2) > best_sim:
				v1 = model[syn1]
				v2 = model[syn2]
				best_sim = model.similarity(syn1, syn2)
	best_sim = 0
	v3, v4 = None, None
	for syn3, _ in d[w3]:
		for syn4, _ in d[w4]:
			if model.similarity(syn3, syn4) > best_sim:
				v3 = model[syn3]
				v4 = model[syn4]
				best_sim = model.similarity(syn3, syn4)
	print 'v1:', v1
	print 'v2:', v2
	print 'v3:', v3
	print 'v4:', v4
	print 'cosine:', cosine(v1-v2, v3-v4)
	return (cosine(v1-v2, v3-v4)+1.0)/2.0