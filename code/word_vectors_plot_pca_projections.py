import gensim, logging
from gensim.models import Word2Vec
from random import shuffle
import numpy
from sklearn.decomposition import PCA
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open('wnet30_dict.txt', 'r')
d = {}
for l in f:
	ls = l.split()
	synsets = []
	for s in ls[1:]:
		synset, count = s.split(':')
		synsets.append((synset, int(count)))
	d[ls[0]] = synsets


model = Word2Vec.load('../wn_model/random_walk_model')

senses = set([s[0] for ls in d.values() for s in ls])
sense_vectors = [(s, model[s]) for s in senses if s in model.vocab]
index = {s[0]: i for i, s in enumerate(sense_vectors)}
sense_vectors = [s[1] for s in sense_vectors]

pca = PCA(n_components=3)
sense_vectors_new = pca.fit_transform(sense_vectors)


# hot, cold, humid; asia, africa, antarctica; school, university, college
# sad, happy, angry; dog, cat, bird; earth, moon, sun
# 
plot_words = ['president', 'prime_minister', 'king', 'oak', 'maple', 'pine', 'electron', 'proton', 'neutron']
c = ['b','b','b','g','g','g','r','r','r']
word_vec = [sense_vectors_new[index[max(d[x], key=lambda d:d[1])[0]]] for x in plot_words]

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
i=0
for x in word_vec:
	ax.plot([0,x[0]], [0,x[1]], [0,x[2]], c[i], linewidth=2.5, label=plot_words[i])
	# plt.annotate(plot_words[i], xy=(x[0], x[1], x[2]), xycoords='data', textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
	i+=1

plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
ax.legend(loc='upper left')
ax.set_xlabel(r'$1^{st}$ principal component')
ax.set_ylabel(r'$2^{nd}$ principal component')
ax.set_zlabel(r'$3^{rd}$ principal component')
ax.set_title('PCA projection of sense-vectors')
plt.show()


# plot_words = ['man', 'woman', 'king', 'queen', 'male', 'female', 'prince', 'princess']
# c = ['b','g','r','m']
# word_vec = [sense_vectors_new[index[max(d[x], key=lambda d:d[1])[0]]] for x in plot_words]

# mpl.rcParams['legend.fontsize'] = 10
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# i=0
# while i < len(word_vec):
# 	x = word_vec[i]
# 	y = word_vec[i+1]
# 	ax.plot([0,x[0]-y[0]], [0,x[1]-y[1]], [0,x[2]-y[2]], c[i/2], linewidth=2.5, label=(plot_words[i]+'-'+plot_words[i+1]))
# 	# plt.annotate(plot_words[i], xy=(x[0], x[1], x[2]), xycoords='data', textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
# 	i+=2

# plt.xlim(-1.0, 1.0)
# plt.ylim(-1.0, 1.0)
# plt.ylim(-1.0, 1.0)
# ax.legend(loc='upper left')
# ax.set_xlabel(r'$1^{st}$ principal component')
# ax.set_ylabel(r'$2^{nd}$ principal component')
# ax.set_zlabel(r'$3^{rd}$ principal component')
# ax.set_title('PCA projection of sense-vectors')
# plt.show()



bank_sense_vectors = [sense_vectors_new[index[x[0]]] for x in d['bank']][:2]
river_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['river']][0]
shore_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['shore']][0]
money_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['money']][0]
credit_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['credit']][1]
loan_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['loan']][0]

word_vec = bank_sense_vectors + [river_sense_vector, shore_sense_vector, money_sense_vector, credit_sense_vector, loan_sense_vector]

plot_words = ['bank_river', 'bank_financial', 'river', 'shore', 'money', 'credit', 'loan']
c = ['b','c','g','g','r','r','r']

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
i=0
for x in word_vec:
	ax.plot([0,x[0]], [0,x[1]], [0,x[2]], c[i], linewidth=2.5, label=plot_words[i])
	i+=1

plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
ax.legend(loc='upper left')
ax.set_xlabel(r'$1^{st}$ principal component')
ax.set_ylabel(r'$2^{nd}$ principal component')
ax.set_zlabel(r'$3^{rd}$ principal component')
ax.set_title('Sense-vectors disambiguate polysemous word - \'bank\'')
plt.show()


wood_sense_vectors = [sense_vectors_new[index[x[0]]] for x in d['wood']][:2]
timber_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['timber']][0]
jungle_sense_vector = [sense_vectors_new[index[x[0]]] for x in d['jungle']][2]

word_vec = wood_sense_vectors + [timber_sense_vector, jungle_sense_vector]

plot_words = ['wood_timber', 'wood_forest', 'timber', 'jungle']
c = ['m','c','r','b']

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
i=0
for x in word_vec:
	ax.plot([0,x[0]], [0,x[1]], [0,x[2]], c[i], linewidth=2.5, label=plot_words[i])
	i+=1

plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
ax.legend(loc='upper left')
ax.set_xlabel(r'$1^{st}$ principal component')
ax.set_ylabel(r'$2^{nd}$ principal component')
ax.set_zlabel(r'$3^{rd}$ principal component')
ax.set_title('Sense-vectors disambiguate polysemous word - \'wood\'')
plt.show()



model = Word2Vec.load('../wn_model/Word2Vec_size_100_window_3_min_count_5_brown_clean_model')

words = set(d.keys())
word_vectors = [(s, model[s]) for s in words if s in model.vocab]
index = {s[0]: i for i, s in enumerate(word_vectors)}
word_vectors = [s[1] for s in word_vectors]

pca = PCA(n_components=3)
word_vectors_new = pca.fit_transform(word_vectors)


plot_words = ['bank', 'river', 'shore', 'money', 'credit', 'loan']
c = ['b','g','g','r','r','r']
word_vec = [word_vectors_new[index[x]] for x in plot_words]

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
i=0
for x in word_vec:
	ax.plot([0,x[0]], [0,x[1]], [0,x[2]], c[i], linewidth=2.5, label=plot_words[i])
	i+=1

plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
ax.legend(loc='upper left')
ax.set_xlabel(r'$1^{st}$ principal component')
ax.set_ylabel(r'$2^{nd}$ principal component')
ax.set_zlabel(r'$3^{rd}$ principal component')
ax.set_title('Word-vectors for polysemous word - \'bank\'')
plt.show()


plot_words = ['wood', 'timber', 'jungle']
c = ['b','g','r']
word_vec = [word_vectors_new[index[x]] for x in plot_words]

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
i=0
for x in word_vec:
	ax.plot([0,x[0]], [0,x[1]], [0,x[2]], c[i], linewidth=2.5, label=plot_words[i])
	i+=1

plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
ax.legend(loc='upper left')
ax.set_xlabel(r'$1^{st}$ principal component')
ax.set_ylabel(r'$2^{nd}$ principal component')
ax.set_zlabel(r'$3^{rd}$ principal component')
ax.set_title('Word-vectors for polysemous word - \'wood\'')
plt.show()

