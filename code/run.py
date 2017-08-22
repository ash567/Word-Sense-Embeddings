import csv
import numpy as np
import gensim
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
import scipy
import sys

# root = "/home/ishu/Documents/academics/7_sem/cs6370_nlp/nlp_project/"
data = "../data/"
results = "../results/"
models = "../models/"
file_name = ""
factor = 1

if len(sys.argv) != 3:
	print 'Format: test-word-pairs-file rating-scale'
	sys.exit()

file_name = sys.argv[1]
factor = int(sys.argv[2])

f = open(data + 'wnet30_dict.txt', 'r')
d = {}
for l in f:
	ls = l.split()
	synsets = []
	for s in ls[1:]:
		synset, count = s.split(':')
		synsets.append((synset, int(count)))
	d[ls[0]] = synsets


def cosine(v1, v2):
	return 1 - scipy.spatial.distance.cosine(v1, v2)
	# return np.dot(v1, v2)/(np.dot(v1, v1) ** 0.5)/(np.dot(v2, v2) ** 0.5)

def relational_similarity(w1, w2, w3, w4, add_n = 0):
	l1 = []
	for syn1, _ in d[w1]:
		for syn2, _ in d[w2]:
			v1 = sense_model[syn1]
			v2 = sense_model[syn2]
			l1.append(v1-v2)
	l2 = []
	for syn3, _ in d[w3]:
		for syn4, _ in d[w4]:
			v3 = sense_model[syn3]
			v4 = sense_model[syn4]
			l2.append(v3-v4)
	ans = -1
	for u in l1:
		for v in l2:
			if np.any(u) and np.any(v):
				ans = max(ans, cosine(u, v))
	return (ans+1)/2


	# best_sim = 0
	# v1, v2 = None, None
	# for syn1, _ in d[w1]:
	# 	for syn2, _ in d[w2]:
	# 		if sense_model.similarity(syn1, syn2) > best_sim:
	# 			v1 = sense_model[syn1]
	# 			v2 = sense_model[syn2]
	# 			best_sim = sense_model.similarity(syn1, syn2)
	# best_sim = 0
	# v3, v4 = None, None
	# for syn3, _ in d[w3]:
	# 	for syn4, _ in d[w4]:
	# 		if sense_model.similarity(syn3, syn4) > best_sim:
	# 			v3 = sense_model[syn3]
	# 			v4 = sense_model[syn4]
	# 			best_sim = sense_model.similarity(syn3, syn4)
	# print 'v1:', v1
	# print 'v2:', v2
	# print 'v3:', v3
	# print 'v4:', v4
	# print 'v1 - v2:', v1 - v2
	# print 'v3 - v3:', v3 - v4
	# print 'cosine:', cosine(v1-v2, v3-v4)
	# return (cosine(v1-v2, v3-v4)+1.0)/2.0


def sense_similarity(w1, w2):
	if w1 not in d or w2 not in d:
		return None
	best_sim = 0
	for syn1, _ in d[w1]:
		for syn2, _ in d[w2]:
			best_sim = max(best_sim, sense_model.similarity(syn1, syn2))
	return best_sim


words = []
human_similarities = []
sense_model_similarities = []
word2vec_model_similarities = []

lmtzr = WordNetLemmatizer()
with open(data + file_name, 'r') as f:
	# data = csv.reader(csvfile, delimiter=',')
	for row in f:
		row = row.split()
		w1 = lmtzr.lemmatize(row[0].lower())
		w2 = lmtzr.lemmatize(row[1].lower())
		words += [(w1, w2)]
		human_similarities += [row[2]]


# TODO:  Use try catch to train the model if not already done

words = words[1:]
human_similarities = human_similarities[1:]
human_similarities = map(float, human_similarities)

total_pairs = len(words)
# TODO: Remove this part

# Remove the words no in word2vec dictionary
word2vec_model = Word2Vec.load(models + 'Word2Vec_size_100_window_3_min_count_5_brown_clean_model')
sense_model = Word2Vec.load(models + 'random_walk_model')



index = [i for i in range(len(words)) if (words[i][0] not in word2vec_model.vocab) or (words[i][1] not in word2vec_model.vocab)]
l = zip(words, human_similarities)
l = [l[i] for i in range(len(l)) if i not in index]
words = [x[0] for x in l]
human_similarities = [x[1] for x in l]


index = [i for i in range(len(words)) if (words[i][0] not in d) or (words[i][1] not in d)]
l = zip(words, human_similarities)
l = [l[i] for i in range(len(l)) if i not in index]
words = [x[0] for x in l]
human_similarities = [x[1] for x in l]

# index = [i for i in range(len(words)) if sense_model_similarities[i] == None]
# for i in index:
# 	print words[i]
tested_pairs = len(words)
print 'total word pairs tested: %d/%d\n\n' %(tested_pairs, total_pairs)

word2vec_model_similarities = [word2vec_model.similarity(word1, word2)*factor if ((word1 in word2vec_model.vocab) and (word2 in word2vec_model.vocab)) else factor/2 for word1, word2 in words]
word2vec_rms = np.sqrt(np.mean(np.square([a-b for (a,b) in zip(human_similarities, word2vec_model_similarities)])))
word2vec_corr = np.corrcoef(human_similarities, word2vec_model_similarities)
word2vec_spear = scipy.stats.spearmanr(human_similarities, word2vec_model_similarities)
print '--word2vec--'
print "rms: %f\ncorrelation: %f\nspearman: %f \n\n" %(word2vec_rms, word2vec_corr[0][1], word2vec_spear.correlation)


sense_model_similarities = [sense_similarity(word1, word2)*factor if (word1 in d and word2 in d) else factor/2 for word1, word2 in words]
sense_rms = np.sqrt(np.mean(np.square([a-b for (a,b) in zip(human_similarities, sense_model_similarities)])))
sense_corr = np.corrcoef(human_similarities, sense_model_similarities)
sense_spear = scipy.stats.spearmanr(human_similarities, sense_model_similarities)
print '--sense--'
print "rms: %f\ncorrelation: %f\nspearman: %f\n" %(sense_rms, sense_corr[0][1], sense_spear.correlation)