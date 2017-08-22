import gensim, logging

# shows the progress during the training

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



# INPUT
# Assumption: Input is clean file

#implement both the types: Sequential and Bulk


root = "/home/ishu/Documents/academics/7_sem/cs6370_nlp/nlp_project/"
data = "data/"
results = "results/"
models = "models/"

input_file = "brown_clean.txt"


training_sentences = open(root + data + input_file, 'r')
# Entire file in ram
input_sentences = [line.lower().split() for line in training_sentences]

# input_sentences = gensim.models.word2vec.LineSentence(root + data + input_file)


# Line read read sequentially from the file one by one
# class MySentences(object):
#     def __init__(self, file_list):
#         self.file_list = file_list
 		
#     def __iter__(self):
#     	for f in file_list:
#     		for line in open(f, "r"):
#                 yield line.split()

# input_sentences = MySentences('/some/directory') # a memory-friendly iterator


# TRAINING
size = 500
window = 3
min_count = 5
workers = 7
alpha = 0.005
min_alpha = 0.0001
# sample = 10.0**(-3 )
seed = 6
sg = 1
hs = 0
# negative = 5
# cbow_mean = 1

iter_no = 20

model_name = "Word2Vec_size_" + str(size) + "_window_" + str(window) + "_min_count_" + str(min_count)

model = gensim.models.word2vec.Word2Vec(iter = 1,
										size = size,
										window = window,
										min_count = min_count,
										workers = workers,
										alpha = alpha,
										min_alpha = min_alpha,
										# sample = sample,
										seed = seed,
										sg = sg,
										# negative = negative,
										# cbow_mean = cbow_mean,
										hs = hs)

model.build_vocab(input_sentences)
model.train(input_sentences)

for i in xrange(iter_no):
	print "\n", "Epoch: " + str(i), "----------", "\n"
	model.train(input_sentences)

# print model['x']
# model.save(root + models + model_name)

model.most_similar(positive = ['queen', 'man'], negative = ['woman'], topn = 10)