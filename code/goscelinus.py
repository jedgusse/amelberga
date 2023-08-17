#!/usr/bin/env
# -*- coding: utf-8 -*-

from binascii import hexlify
from collections import OrderedDict
from cltk.prosody.latin.syllabifier import Syllabifier
from cltk.stem.lemma import LemmaReplacer
from collatex import *
from collections import Counter, namedtuple
from itertools import combinations, compress, product
from lexical_diversity import lex_div as ld
from matplotlib import colors
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from operator import itemgetter
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn import svm, preprocessing
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, FunctionTransformer, LabelBinarizer, MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from string import punctuation
from tqdm import tqdm
from tqdm import trange
import argparse
import colorsys
import csv
import glob
import heapq
import matplotlib.font_manager as fm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import operator
import os
import pandas as pd
import pickle
import random
import re
import sys
import warnings

"""
PARAMETERS
"""
sample_len = 1400 # length of sample / segment
n_feats = 500 # number of features taken in to account
ignore_words = [] # Words in list will be ignored in analysis
list_of_function_words = open('/Users/...').read().split() # Loads manual list of function words

"""
GENERAL CLASSES AND FUNCTIONS
"""
def enclitic_split(input_str):
	# Feed string, returns lowercased text with split enclitic -que
	que_list = open("/Users/jedgusse/compstyl/params/que_list.txt").read().split()
	spaced_text = []
	for word in input_str.split():
		word = "".join([char for char in word if char not in punctuation]).lower()
		if word[-3:] == 'que' and word not in que_list:
			word = word.replace('que','') + ' que'
		spaced_text.append(word)
	spaced_text = " ".join(spaced_text)
	return spaced_text

def words_and_bigrams(text):
	words = re.findall(r'\w{1,}', text)
	for w in words:
		if w not in stop_words:
			yield w.lower()
		for i in range(len(words) - 2):
			if ' '.join(words[i:i+2]) not in stop_words:
				yield ' '.join(words[i:i+2]).lower()

def to_dense(X):
		X = X.todense()
		X = np.nan_to_num(X)
		return X

def deltavectorizer(X):
		# "An expression of pure difference is what we need"
		#  Burrows' Delta -> Absolute Z-scores
		X = np.abs(stats.zscore(X))
		X = np.nan_to_num(X)
		return X

def most_common(lst):
	return max(set(lst), key=lst.count)

def align_yaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	_, y1 = ax1.transData.transform((0, v1))
	_, y2 = ax2.transData.transform((0, v2))
	inv = ax2.transData.inverted()
	_, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
	miny, maxy = ax2.get_ylim()
	ax2.set_ylim(miny+dy, maxy+dy)

def align_xaxis(ax1, v1, ax2, v2):
	"""adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
	x1, _ = ax1.transData.transform((v1, 0))
	x2, _ = ax2.transData.transform((v2, 0))
	inv = ax2.transData.inverted()
	dx, _ = inv.transform((0, 0)) - inv.transform((x1-x2, 0))
	minx, maxx = ax2.get_xlim()
	ax2.set_xlim(minx+dx, maxx+dx)

def change_intensity(color, amount=0.5):
	"""
	Lightens the given color by multiplying (1-luminosity) by the given amount.
	Input can be matplotlib color string, hex string, or RGB tuple.

	Examples:
	>> change_intensity('g', 0.3)
	>> change_intensity('#F034A3', 0.6)
	>> change_intensity((.3,.55,.1), 0.5)
	https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

	setting an amount < 1 lightens
	setting an amount > 1 darkens too

	"""
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def remove_diacritics(token):
	normalized_vowels = {'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u', 'y': 'y', 'ŷ': 'y', 'ȳ': 'y', 'â': 'a', 'ă': 'a', 'ā': 'a', 'ê': 'e', 'ĕ': 'e', 'ē': 'e', 'î': 'i', 'ĭ': 'i', 'ī': 'i',  'ô': 'o', 'ŏ': 'o', 'ō': 'o', 'û': 'u', 'ŭ': 'u', 'ū': 'u'}
	rhyme_match = token[1:] # ['ϑmā', 'ϑmă']
	normalized = []
	for match in rhyme_match:
		normalized_match = []
		for let in match:
			if let in normalized_vowels.keys(): # breves and macrons get smoothened out
				normalized_match.append(normalized_vowels[let])
			else:
				normalized_match.append(let)
		
		normalized.append(''.join(normalized_match))
	normalized.insert(0, token[0]) # reinserts the rhymetype (e.g. 'asϑuς')

	return normalized

def syllab_matcher(syllabs):

	fricatives = ['v', 'f']
	plosives = ['b', 'p']
	dentals = ['t', 'd']
	consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'l', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x']
	vowels = ['a', 'e', 'i', 'o', 'u', 'y']
	macrons = ['ā', 'ē', 'ī', 'ō', 'ū', 'ȳ']
	breves = ['ă', 'ĕ', 'ĭ', 'ŏ', 'ŭ']
	ambiguous_quantities = ['â', 'ê', 'î', 'ô', 'û', 'ŷ']
	normalized_vowels = {'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u', 'y': 'y', 'ŷ': 'y', 'ȳ': 'y', 'â': 'a', 'ă': 'a', 'ā': 'a', 'ê': 'e', 'ĕ': 'e', 'ē': 'e', 'î': 'i', 'ĭ': 'i', 'ī': 'i',  'ô': 'o', 'ŏ': 'o', 'ō': 'o', 'û': 'u', 'ŭ': 'u', 'ū': 'u'}
	all_vowels = vowels + macrons + breves + ambiguous_quantities
	all_specials = macrons + breves + ambiguous_quantities

	letters = [[letter for letter in syllab] for syllab in syllabs] # [['g', 'ī', 's'], ['t', 'r', 'ă']]
	# establish length of longest syllable and detect rhyme from furthest right letter onward 
	n_iters = np.max([len(syllab) for syllab in letters])
	iterations = list(range(1, n_iters+1))
	iterations = [-abs(val) for val in iterations] # [-1, -2, -3]

	# return indices of correspondence
	correspondents = []
	for idx in iterations:
		letter_pair = [] # eventually something like this: ['a', 'ă']
		for letter in letters:
			try:
				focus_letter = letter[idx]
			except IndexError:
				focus_letter = '-' # compared syllables are unequal length
			letter_pair.append(focus_letter)
		if len(set(letter_pair)) == 1: # match detected
			correspondents.append(idx)
		else:
			normalized_letter_pair = [] # correspondence between breves - macrons gets exceptionalized
			for letter in letter_pair:
				if letter in normalized_vowels.keys(): # breves and macrons get smoothened out
					normalized_letter_pair.append(normalized_vowels[letter])
				else: # all consonants and other letters stay as they are 
					normalized_letter_pair.append(letter)
			if len(set(normalized_letter_pair)) == 1: # match detected
				correspondents.append(idx)
	
	return correspondents

class DataReader:
	"""
	Parameters 
	----------
	folder_location: location of .txt files in directory
	sample_len: declare sample length

	Returns
	-------
	Lists
	authors = [A, A, B, B, ...]
	titles = [A, A, B, B, ...]
	texts = [s, s, s, s, ...] # strings
	"""

	def __init__(self, folder_location, sample_len):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def fit(self, shingling, shingle_titles):

		authors = []
		titles = []
		texts = []

		if shingling == 'n':
		
			for filename in glob.glob(self.folder_location + "/*"):
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]

				bulk = []
				text = open(filename).read()

				for word in text.strip().split():
					word = re.sub('\d+', '', word) # escape digits
					word = re.sub('[%s]' % re.escape(punctuation), '', word) # escape punctuation
					word = word.lower() # convert upper to lowercase
					bulk.append(word)

				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]
				bulk = [bulk[i:i+self.sample_len] for i \
					in range(0, len(bulk), self.sample_len)]
				for index, sample in enumerate(bulk):
					if len(sample) == self.sample_len:
						authors.append(author)
						titles.append(title + "_{}".format(str(index + 1)))
						texts.append(" ".join(sample))

		# titles which should be shingled can be fed
		if shingling == 'y':
			
			step_size = 10

			for filename in glob.glob(self.folder_location + "/*"):
				
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]
				text = open(filename).read()

				text = re.sub('[%s]' % re.escape(punctuation), '', text) # Escape punctuation and make characters lowercase
				text = re.sub('\d+', '', text)
				text = text.lower().split()

				if title in shingle_titles:

					steps = np.arange(0, len(text), step_size)
					step_ranges = []
					data = {}
					for each_begin in steps:
						key = '{}-{}-{}'.format(title, str(each_begin), str(each_begin + sample_len))
						sample_range = range(each_begin, each_begin + sample_len)
						step_ranges.append(sample_range)
						sample = []
						for index, word in enumerate(text):
							if index in sample_range:
								sample.append(word)
						if len(sample) == sample_len:
							data[key] = sample

					for key, sample in data.items():
						authors.append(author)
						titles.append(key)
						texts.append(" ".join(sample))

				else:
					# Safety measure against empty strings in samples
					bulk = [word for word in text if word != ""]
					bulk = [bulk[i:i + sample_len] for i \
						in range(0, len(bulk), sample_len)]
					for index, sample in enumerate(bulk):
						if len(sample) == sample_len:
							authors.append(author)
							titles.append(title + "_{}".format(str(index + 1)))
							texts.append(" ".join(sample))

		return authors, titles, texts


class RhymeDataReader:
	"""
	Parameters 
	----------
	folder_location: location of .txt files in directory
	sample_len: declare sample length

	Returns
	-------
	Lists
	authors = [A, A, B, B, ...]
	titles = [A, A, B, B, ...]
	texts = [s, s, s, s, ...] # strings
	"""

	def __init__(self, folder_location, sample_len):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def fit(self, shingling, shingle_titles):

		authors = []
		titles = []
		texts = []

		if shingling == 'n':
		
			for filename in glob.glob(self.folder_location + "/*"):
				author = filename.split("/")[-1].split(".")[0].split("_")[0]
				title = filename.split("/")[-1].split(".")[0].split("_")[1]

				bulk = []
				for line in open(filename):
					rhyme_match = str(line.strip())
					bulk.append(rhyme_match)

				# Safety measure against empty strings in samples
				bulk = [word for word in bulk if word != ""]
				bulk = [bulk[i:i+self.sample_len] for i \
					in range(0, len(bulk), self.sample_len)]
				for index, sample in enumerate(bulk):
					if len(sample) == self.sample_len:
						authors.append(author)
						titles.append(title + "_{}".format(str(index + 1)))
						texts.append(" ".join(sample))

		return authors, titles, texts

class Vectorizer:
	"""
	Independent class to vectorize texts.

	Parameters
	---------
	"""

	def __init__(self, texts, stop_words, n_feats, feat_scaling, analyzer, vocab):
		self.texts = texts
		self.stop_words = stop_words
		self.n_feats = n_feats
		self.feat_scaling = feat_scaling
		self.analyzer = analyzer
		self.vocab = vocab
		self.norm_dict = {'delta': FunctionTransformer(deltavectorizer), 
						  'normalizer': Normalizer(),
						  'standard_scaler': StandardScaler(),
						  'minmaxscaler': MinMaxScaler()}

	# Raw Vectorization

	def raw(self):

		# Text vectorization; array reversed to order of highest frequency
		# Vectorizer takes a list of strings

		# Define fed-in analyzer
		ngram_range = None
		if self.analyzer == 'char':
			ngram_range = ((4,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		"""option where only words from vocab are taken into account"""
		model = CountVectorizer(stop_words=self.stop_words, 
								max_features=self.n_feats,
								analyzer=self.analyzer,
								vocabulary=self.vocab,
								ngram_range=ngram_range)


		doc_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(doc_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			doc_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			doc_features = model.get_feature_names()
			doc_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			doc_features = doc_features[:self.n_feats]

		new_X = []
		for feat in doc_features:
			for ft, vec in zip(model.get_feature_names(), doc_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		doc_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			doc_vectors = scaling_model.fit_transform(doc_vectors)

		return doc_vectors, doc_features, scaling_model

	# Term-Frequency Inverse Document Frequency Vectorization

	def tfidf(self, smoothing):

		# Define fed-in analyzer
		ngram_range = None
		stop_words = self.stop_words
		if self.analyzer == 'char':
			ngram_range = ((4,4))
		elif self.analyzer == 'word':
			ngram_range = ((1,1))

		model = TfidfVectorizer(stop_words=self.stop_words, 
								max_features=self.n_feats,
								analyzer=self.analyzer,
								vocabulary=self.vocab,
								ngram_range=ngram_range)

		tfidf_vectors = model.fit_transform(self.texts).toarray()
		corpus_vector = np.ravel(np.sum(tfidf_vectors, axis=0))
		
		""" ||| Input vocabulary retains original order, 
		new vocabulary is ordered in terms of frequency |||"""
		if self.vocab == None:
			features = model.get_feature_names()
			tfidf_features = [x for (y,x) in sorted(zip(corpus_vector, features), reverse=True)]
		else:
			"""if a vocabulary is given, sort it in terms of freq nevertheless"""
			features = self.vocab
			tfidf_features = model.get_feature_names()
			tfidf_features = [feat for (freq, feat) in sorted(zip(corpus_vector, features), reverse=True)]
			"""only retain max number of n feats"""
			tfidf_features = tfidf_features[:self.n_feats]

		new_X = []
		for feat in tfidf_features:
			for ft, vec in zip(model.get_feature_names(), tfidf_vectors.transpose()):
				if feat == ft: 
					new_X.append(vec)
		new_X = np.array(new_X).transpose()
		tfidf_vectors = new_X

		if self.feat_scaling == False:
			scaling_model = None
			pass
		else:
			scaling_model = self.norm_dict[self.feat_scaling]
			tfidf_vectors = scaling_model.fit_transform(tfidf_vectors)
			
		return tfidf_vectors, tfidf_features, scaling_model

class SVM_benchmarking:
	"""
	Class that runs a parameter search.

	Parameters
	---------
	folder_location = location of .txt files
	
	Returns
	-------
	optimal_sample_len
	optimal_feature_type
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location

	def go(self):
		results_file = open('/Users/...')

		# chi2_best_fwords = ['ac', 'tamquam', 'hanc', 'immo', 'siquidem', 'absque', 'idem', 'uti', 'eius', 'postea', 'eorum', 'eo', 'quodque', 'utrum', 'primum', 'eis', 'enim', 'is', 'quodam', 'tuis', 'aliquid', 'quamquam', 'mi', 'eos', 'quidam', 'scilicet', 'ei', 'autem', 'mox', 'illuc']
		function_words_only = open('/Users/...').read().split()
		# sample_len_loop = list(range(50, 1350, 25))
		sample_len_loop = [500]
		# feat_type_loop = ['raw_fwords','raw_MFW','raw_4grams','tfidf_fwords','tfidf_MFW','tfidf_4grams']
		feat_type_loop = ['tfidf_fwords']
		c_options = [1] # 10, 100 and 1000 also possible, but on average scores are worse

		for feat_type in feat_type_loop:
			# vector length has to differ according to feature type (since there are no 1,000 function words)
			if feat_type.split('_')[-1] == 'fwords':
				# feat_n_loop = [50, 150, 250, 300]
				feat_n_loop = [250]
			else:
				feat_n_loop = [250, 500, 750, 1000]
			
			for n_feats in feat_n_loop:
				for sample_len in sample_len_loop:
					# Leave One Out cross-validation
					"""
					PREPROCESSING
					-------------
					"""
					# Load training files
					# The val_1 and val_2 pass True or False arguments to the sampling method
					# authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
					authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
					invalid_words = []
						
					# Number of splits is based on number of samples, so only possible afterwards
					# Minimum is 2
					n_cval_splits = 10

					# Try both stratified cross-validation as 'normal' KFold cross-validation.
					# Stratification has already taken place with random sampling
					cv_types = []
					cv_types.append(StratifiedKFold(n_splits=n_cval_splits))

					"""
					INSTANTIATE VECTORIZER
					"""
					if feat_type == 'raw_MFW': 
						vectorizer = CountVectorizer(stop_words=invalid_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)
					elif feat_type == 'tfidf_MFW': 
						vectorizer = TfidfVectorizer(stop_words=invalid_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)
					elif feat_type == 'raw_fwords':
						"""
						All content words of corpus are rendered invalid
						and fed in the model as stop_words
						"""
						stop_words = [t.split() for t in texts]
						stop_words = sum(stop_words, [])
						stop_words = [w for w in stop_words if w not in list_of_function_words]
						stop_words = set(stop_words)
						stop_words = list(stop_words)

						"""
						----
						"""
						vectorizer = CountVectorizer(stop_words=stop_words,
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats)

					elif feat_type == 'tfidf_fwords': 
						"""
						Low-frequency function words gain higher weight
						Filters out words that are not function words
						"""
						stop_words = [t.split() for t in texts]
						stop_words = sum(stop_words, [])
						stop_words = [w for w in stop_words if w not in list_of_function_words]
						stop_words = set(stop_words)
						stop_words = list(stop_words)
						"""
						----
						"""
						vectorizer = TfidfVectorizer(stop_words=stop_words, 
													analyzer='word', 
													ngram_range=(1, 1),
													max_features=n_feats) # feed in best-scoring chi2 fwords

					elif feat_type == 'raw_4grams': 
						vectorizer = CountVectorizer(stop_words=invalid_words, 
													analyzer='char', 
													ngram_range=(4, 4),
													max_features=n_feats)

					elif feat_type == 'tfidf_4grams': 
						vectorizer = TfidfVectorizer(stop_words=invalid_words, 
													analyzer='char', 
													ngram_range=(4, 4),
													max_features=n_feats)
					
					"""
					ENCODING X_TRAIN, x_test AND Y_TRAIN, y_test
					--------------------------------------------
			 		"""
					# Arranging dictionary where title is mapped to encoded label
					# Ultimately yields Y_train

					label_dict = {}
					inverse_label = {}
					for title in authors: 
						label_dict[title.split('_')[0]] = 0 
					for i, key in zip(range(len(label_dict)), label_dict.keys()):
						label_dict[key] = i
						inverse_label[i] = key

					"""
					TRAINING

					Step 1: input string is vectorized
						e.g. '... et quam fulgentes estis in summo sole ...'
					Step 2: to_dense = make sparse into dense matrix
					Step 3: feature scaling = normalize frequencies to chosen standard
					Step 4: reduce dimensionality by performing feature selection
					Step 5: choose type of classifier with specific decision function

					"""
					# Map Y_train to label_dict

					Y_train = []
					for title in authors:
						label = label_dict[title.split('_')[0]]
						Y_train.append(label)

					Y_train = np.array(Y_train)
					X_train = texts

					# DECLARING GRID, TRAINING
					# ------------------------
					"""
					Put this block of code in comment when skipping training and loading model
					Explicit mention labels=Y_train in order for next arg average to work
					average='macro' denotes that precision-recall scoring (in principle always binary) 
					needs to be averaged for multi-class problem
					"""

					pipe = Pipeline(
						[('vectorizer', vectorizer),
						 ('to_dense', FunctionTransformer(to_dense, accept_sparse=True)),
						 ('feature_scaling', StandardScaler()),
						 # ('reduce_dim', SelectKBest(mutual_info_regression)),
						 ('classifier', svm.SVC(probability=True))])

					# c_options = [c_parameter]
					# n_features_options = [n_feats]
					kernel_options = ['linear']

					param_grid = [	
						{
							'vectorizer': [vectorizer],
							'feature_scaling': [StandardScaler()],
							# 'reduce_dim': [SelectKBest(mutual_info_regression)],
							# 'reduce_dim__k': n_selected_feats,
							'classifier__C': c_options,
							'classifier__kernel': kernel_options,
						},
					]

					# Change this parameter according to preferred high scoring metric
					refit = 'accuracy_score'

					for cv in cv_types:

						print(":::{} as feature type, {} as number of features ::::".format(feat_type, str(n_feats)))
						grid = GridSearchCV(pipe, cv=cv, n_jobs=9, param_grid=param_grid,
											scoring={
										 		'precision_score': make_scorer(precision_score, labels=Y_train, \
										 														 average='macro'),
												'recall_score': make_scorer(recall_score, labels=Y_train, \
												 										  average='macro'),
												'f1_score': make_scorer(f1_score, labels=Y_train, \
												 								  average='macro'),
												'accuracy_score': make_scorer(accuracy_score),},
											refit=refit, 
											# Refit determines which scoring method weighs through in model selection
											verbose=True
											# Verbosity level: amount of info during training
											) 

						# Get best model & parameters
						# Save model locally
						grid.fit(X_train, Y_train)
						model = grid.best_estimator_

						# Safety buffer: to avoid errors in code
						vectorizer = model.named_steps['vectorizer']
						classifier = model.named_steps['classifier']
						features = vectorizer.get_feature_names()
						print(features)
						best_c_param = classifier.get_params()['C']
						# features_booleans = grid.best_params_['reduce_dim'].get_support()
						# grid_features = list(compress(features, features_booleans))

						if len(features) != n_feats:
							sys.exit("ERROR: Inconsistent number of features: {} against {}".format(str(n_feats),str(len(features))))

						model_name = '{}-{}feats-{}w-c{}-model'.format(feat_type, str(n_feats), str(sample_len), str(best_c_param))
						model_location = '/Users/...'.format(feat_type, str(n_feats), str(sample_len), str(best_c_param))
						pickle.dump(grid, open(model_location, 'wb'))

						accuracy = grid.cv_results_['mean_test_accuracy_score'][0]
						precision = grid.cv_results_['mean_test_precision_score'][0]
						recall = grid.cv_results_['mean_test_recall_score'][0]
						f1 = grid.cv_results_['mean_test_f1_score'][0]

						results_file.write(model_name + '\t' + str(accuracy))
						results_file.write('\n')

class Plot_Lowess:
	"""
	Class that outputs accuracy in function of a variable
	with fitted lowess line

	Parameters
	----------
	results_location: opens file containing model results
	
	Returns
	-------
	.pdf figure
	"""

	def __init__(self, results_location):
		self.results_location = results_location

	def plot(self):

		results_location = open('')
		sample_len_loop = list(range(50, 1350, 25))

		x = sample_len_loop
		# actual accuracies!
		y = [0.6302142702546714, 0.6462571662571663, 0.6509753298909925, 0.7146766169154228, 0.7163311688311688, 0.7263297872340425, 0.8035423925667828, 0.8222222222222222, 0.7902462121212122, 0.8275268817204303, 0.8359788359788359, 0.8235384615384616, 0.8327898550724637, 0.870995670995671, 0.8609523809523809, 0.8868421052631579, 0.8783625730994151, 0.9058823529411765, 0.8889705882352942, 0.8970833333333333, 0.9466666666666667, 0.8857142857142858, 0.9417582417582417, 0.9141025641025641, 0.9365384615384615, 0.9416666666666667, 0.9393939393939394, 0.9181818181818183, 0.9245454545454546, 0.9718181818181819, 0.9277777777777778, 0.95, 0.9477777777777778, 0.9788888888888888, 0.9666666666666666, 0.9666666666666666, 0.9277777777777778, 0.9416666666666667, 0.95, 0.95, 0.9625, 0.9625, 0.975, 0.9857142857142858, 0.9857142857142858, 0.9857142857142858, 0.9714285714285715, 0.9571428571428571, 0.9523809523809523, 0.9833333333333334, 0.9833333333333334, 0.9666666666666668]

		fig = plt.figure(figsize=(6,2))
		ax = fig.add_subplot(111)

		ys0 = lowess(y, x)
		lowess_x0 = ys0[:,0]
		lowess_y0 = ys0[:,1]

		for p1, p2 in zip(x, y):
			ax.scatter(p1, p2, marker='o', color='w', s=10, edgecolors='k', linewidth=0.5)

		ax.plot(lowess_x0, lowess_y0, color='k', linewidth=0.5, markersize=1.5, linestyle='--')

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['Bodoni 72'] # Font of Revue Mabillon

		ax.set_xlabel('Sample length')
		ax.set_ylabel('Accuracy')

		for tick in ax.xaxis.get_major_ticks():
			tick.label.set_fontsize(7)
		for tick in ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(7)

		# Despine
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_visible(True)

		plt.tight_layout()
		plt.show()

		fig.savefig("", \
					transparent=True, format='pdf')

class PrinCompAnal:

	""" |--- Principal Components Analysis ---|
		::: Plots PCA Plot ::: """

	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot(folder_location, sample_len):

		# invalid_words = ['illarum', 'illas', 'earum', 'he', 'eas']
		# invalid_words = ['sic', 'ipsius', 'qua', 'quoniam', 'ac', 'ergo', 'id', 'tamen', 'namque', 'quasi' , 'quadam', 'ubi', 'sub', 'denique', 'simul', 'tum', 'quomodo', 'usque', 'unde' , 'nisi', 'ideo', 'dehinc', 'nam', 'modo', 'licet', 'inde', 'tuum', 'ipso', 'illuc' , 'ibique', 'hinc', 'tui', 'tuam', 'quidem', 'quendam', 'quandam', 'tali' , 'quoddam', 'plerisque', 'illic', 'donec', 'queque', 'primum', 'huc', 'uidelicet' , 'uester', 'ubique', 'tuorum', 'solus', 'semel']
		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[''])
		# authors, titles, texts = RhymeDataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[''])
		invalid_words = []

		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='standard_scaler',
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		# Uncomment for visualizing rhymes on top of PCA plot
		replace_pattern  = re.compile('ϑ') # strange sign to mask '-'
		replace_pattern_2  = re.compile('ççç') # strange sign to mask ' : '
		replace_pattern_3 = re.compile('ς') # strange sign to mask '_'
		features = [re.sub(replace_pattern, '-', feat) for feat in features]
		features = [re.sub(replace_pattern_2, ' : ', feat) for feat in features]
		features = [re.sub(replace_pattern_3, '_', feat) for feat in features]

		pca = PCA(n_components=3)
		
		X_bar = pca.fit_transform(X)
		var_exp = pca.explained_variance_ratio_

		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		var_pc3 = np.round(var_exp[2]*100, decimals=2)
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)

		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(features, loadings[:,0]), \
								  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(features, loadings[:,1]), \
						   		  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p3 = sorted(zip(features, loadings[:,2]), \
						   		  key=lambda tup: tup[1], reverse=True)


		print("Explained variance: ", explained_variance)
		print("Number of words: ", len(features))
		print("Sample length : ", sample_len)
		
		# Line that calls font (for doctoral thesis lay-out)
		# Custom fonts should be added to the matplotlibrc parameter .json files (fontlist-v310.json): cd '/Users/...'.
		# Make sure the font is in the font "library" (not just font book!)
		# Also: you would want to change the 'name : xxx' entry in the fontlist-v310.json file.
		# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib

		# rcParams['font.family'] = 'sans-serif'
		# rcParams['font.family'] = ['Arno Pro']
		rcParams['font.family'] = 'Times New Roman'

		customized_colors = {'Eadmerus': '#DF859F',
							 'Encomiast': '#3F9EA3',
							 'Folcardus': '#9ECB8B',
							 'Goscelinus': '#90A4DB',
							 'Hermannus-archidiaconus': '#3F9EA3',
							 'B': '#4A94BA',
							 'Bovo-Sithiensis': '#4A94BA',
							 'Byrhtferth-of-Ramsey': '#BC6F36',
							 'Lantfred-of-Winchester': '#C79491',
							 'Wulfstan-of-Winchester': '#A2D891',
							 'Vita-et-Inventio-Amelbergae': '#F5B97F',
							 'Vita-Ædwardi': 'r',
							 'VHerRein': '#C4D200',
							 'Heriger-of-Lobbes': '#3CA56A',
							 'The-Destruction-of-Troy': 'r',
							 'In-Cath-Catharda': 'b',
							 'VAmelb': 'r',
							 'Theodericus-Trudonensis': 'r'}
		replace_dict = {'Historia-minor-Augustini': 'HAug-min',
						'Inventio-Yvonis[PL+Macray]': 'InvYv',
						'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 'Libellus',
						'Liber-confortatorius': 'LC',
						'Miracula-Augustini-maior': 'MAug-mai',
						'Translatio-Augustini-et-aliorum-sanctorum': 'TAug',
						'Translatio-Mildrethe': 'TMild',
						'Vita-Augustini': 'VAug',
						'Vita-Edithae-no-verse': 'VEdith',
						'Vita-Mildrethe': 'VMild',
						'Vita-Vulfilda': 'VVulf',
						'Vita-Wlsini(amended-Love)': 'VWls',
						'Vita-de-Iusto-archiepiscopo': 'VIust',
						'Vita-et-Miracula-Æthelburge': 'VÆthelb',
						'In-natale-S-Edwoldi': 'NatEdw',
						'Lectiones-Eormenhilde': 'LEorm',
						'Lectiones-Sexburge': 'LSex',
						'Lectiones-de-Hildelitha': 'LHild',
						'Miracula-Aetheldrethe': 'MÆtheld',
						'Miracula-S-Eadmundi-regis-et-martyris': 'MEadm',
						'Miracula-Wihtburge': 'MWiht',
						'Passio-Eadwardi-regis-et-martyris': 'PEadw',
						'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 'TÆthelb-mai',
						'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 'TÆthelb-min',
						'Translatio-Wulfildae': 'TWulf',
						'Translatio-Yvonis': 'TYv',
						'Visio-Alvivae': 'VisAlv',
						'Vita-Aetheldrethe-recensio-C': 'VÆtheld-c',
						'Vita-Aetheldrethe-recensio-D': 'VÆtheld-d',
						'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 'VAmel',
						'Vita-Kenelmi-brevior': 'VKen-brev',
						'Vita-Kenelmi': 'VKen',
						'Vita-Letardi': 'VLet',
						'Vita-Milburge-ed-Love': 'VMilb',
						'Vita-Milburge': 'VMilb',
						'Vita-Sexburge': 'VSex',
						'Vita-Werburge': 'VWer',
						'Vita-Wihtburge': 'VWiht',
						'Vita-et-Inventio-Amelbergae': 'VAmel',
						'Vita-Ædwardi copy': 'VÆdwR',
						'Vita-Ædwardi-no-verse': 'VÆdwR',
						'Vita-S-Bertini': 'VBert',
						'Vita-sancti-Johannis-Beverlacensis': 'VIoan',
						'Vita-Botulphi': 'VBot',
						'merger-Thorney': 'Thorn'}

		fig = plt.figure(figsize=(5,3.2))
		ax = fig.add_subplot(111, projection='3d')
		
		x1, x2, x3 = X_bar[:,0], X_bar[:,1], X_bar[:,2]

		# Plot loadings in 3D
		l1, l2, l3 = loadings[:,0], loadings[:,1], loadings[:,2]

		scaler_one = MinMaxScaler(feature_range=(min(x1), max(x1)))
		scaler_two = MinMaxScaler(feature_range=(min(x2), max(x2)))
		scaler_three = MinMaxScaler(feature_range=(min(x3), max(x3)))

		realigned_l1 = scaler_one.fit_transform(l1.reshape(-1, 1)).flatten()
		realigned_l2 = scaler_two.fit_transform(l2.reshape(-1, 1)).flatten()
		realigned_l3 = scaler_three.fit_transform(l3.reshape(-1, 1)).flatten()
			
		# Makes the opacity of plotted features work
		abs_l1 = np.abs(l1)
		abs_l2 = np.abs(l2)
		abs_l3 = np.abs(l3)
		normalized_l1 = (abs_l1-min(abs_l1))/(max(abs_l1)-min(abs_l1))
		normalized_l2 = (abs_l2-min(abs_l2))/(max(abs_l2)-min(abs_l2))
		normalized_l3 = (abs_l3-min(abs_l3))/(max(abs_l3)-min(abs_l3))

		normalized_vocab_weights_p1 = sorted(zip(features, normalized_l1), \
								  key=lambda tup: tup[1], reverse=True)
		normalized_vocab_weights_p2 = sorted(zip(features, normalized_l2), \
						   		  key=lambda tup: tup[1], reverse=True)
		normalized_vocab_weights_p3 = sorted(zip(features, normalized_l3), \
						   		  key=lambda tup: tup[1], reverse=True)

		# Each feature's rank of importance on each of the PC's is calculated
		# Normalized by importance of PC
		d = {}
		for (feat, weight) in normalized_vocab_weights_p1:
			d[feat] = []
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p1):
			d[feat].append(idx * var_pc1)
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p2):
			d[feat].append(idx * var_pc2)
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p3):
			d[feat].append(idx * var_pc3)

		n_top_discriminants = 20 # adjust to visualize fewer or more discriminants
		best_discriminants = sorted([[feat, np.average(ranks)] for [feat, ranks] in d.items()], key = lambda x: x[1])
		top_discriminants = [i[0] for i in best_discriminants][:n_top_discriminants]

		# Scatterplot of datapoints
		for index, (p1, p2, p3, a, title) in enumerate(zip(x1, x2, x3, authors, titles)):
			markersymbol = 'o'
			markersize = 20

			full_title = title.split('_')[0]
			sample_number = title.split('_')[-1]
			abbrev = replace_dict[full_title] + '-' + sample_number

			if full_title in ['Vita-Milburge-ed-Love']:
				# Uncomment for data points
				ax.scatter(p1, p2, p3, marker='o', color='#FFE79B', \
						   s=markersize, zorder=3, alpha=1, edgecolors='k', linewidth=0.3)
				# Cloudy
				ax.scatter(p1, p2, p3, marker='o', color='#FFE79B', \
						   s=500, zorder=1, alpha=0.2)
			# elif full_title in ['Vita-Werburge']:
			# 	# Uncomment for data points
			# 	ax.scatter(p1, p2, p3, marker='o', color='#AEE9CF', \
			# 			   s=markersize, zorder=3, alpha=1, edgecolors='k', linewidth=0.3)
			# 	# Cloudy
			# 	ax.scatter(p1, p2, p3, marker='o', color='#AEE9CF', \
			# 			   s=500, zorder=1, alpha=0.2)
			# 	# Black and white
			# 	ax.scatter(p1, p2, p3, marker='^', color='w', \
			# 			   edgecolors='k', linewidth=0.3, s=markersize, zorder=1, alpha=1)
			elif full_title in ['Vita-et-Inventio-Amelbergae']:
				# Uncomment for data points
				# ax.scatter(p1, p2, p3, marker='o', color=customized_colors[a], \
				# 		   s=markersize, zorder=3, alpha=1, edgecolors='k', linewidth=0.3)
				# Cloudy
				# ax.scatter(p1, p2, p3, marker='o', color=customized_colors[a], \
				# 		   s=500, zorder=1, alpha=0.2)
				# Black and white
				ax.scatter(p1, p2, p3, marker='o', color='k', \
						   s=markersize, zorder=1, alpha=1)
				# ax.text(p1, p2, p3, abbrev, color='k', ha='center', va="center", fontdict={'size': 5}, zorder=10000, alpha=1)
			elif a == 'Goscelinus':
				# ax.scatter(p1, p2, p3, marker='o', color='#FFAEAE', s=markersize, zorder=3, alpha=1, edgecolors='k', linewidth=0.3)
				# ax.scatter(p1, p2, p3, marker='o', color='#FFAEAE', s=500, zorder=1, alpha=0.2)
				ax.scatter(p1, p2, p3, marker='^', color='k', s=markersize, zorder=1, alpha=1)
				# ax.text(p1, p2, p3, sample_number, color='k', ha='center', va="center", fontdict={'size': 5}, zorder=10000, alpha=1)
			elif a == 'Folcardus':
				ax.scatter(p1, p2, p3, marker='x', color='k', s=markersize, zorder=1, alpha=1)

		# Plot features
		for x, y, z, opac_l1, opac_l2, opac_l3, feat, in zip(realigned_l1, realigned_l2, realigned_l3, normalized_l1, normalized_l2, normalized_l3, features):
			total_opac = (opac_l1 + opac_l2 + opac_l3)/3
			if feat in top_discriminants:
				ax.text(x, y, z, feat, color='k', ha='center', va="center", fontdict={'size': 17*total_opac}, zorder=10000, alpha=total_opac)

		# Important to adjust margins first when function words fall outside plot
		# This is due to the axes aligning (def align).
		# ax2.margins(x=0.15, y=0.15)

		ax.set_xlabel('PC 1: {}%'.format(var_pc1))
		ax.set_ylabel('PC 2: {}%'.format(var_pc2))
		ax.set_zlabel('PC 3: {}%'.format(var_pc3))

		plt.tight_layout()
		plt.show()

		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/pca-.png", dpi=300, transparent=True, format='png', bbox_inches='tight')

class Measure_Lexical_Diversity:
	"""
	Class that measures lexical diversity of sentences in corpus.

	Parameters
	---------
	folder_location = location of .txt files
	
	Returns
	-------
	low_to_high_ld = [s, s, s, s, ...]  # list of sents from low to high diversity
	scores = [fl, fl, fl, fl, ...] # list of sorted scores from low to high

	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def go(self):
		
		scores = []
		sents = []

		for filename in glob.glob(folder_location + "/*"):
			author = filename.split("/")[-1].split(".")[0].split("_")[0]
			title = filename.split("/")[-1].split(".")[0].split("_")[1]
			text = open(filename).read().strip()
			text = re.split('\.|\?|\!', text)

			for sent in text:
				tokenized_sent = ld.tokenize(sent)
				if len(tokenized_sent) in range(50,70):

					masked_sent = [w for w in tokenized_sent if w in list_of_function_words]
					print(masked_sent)
					ttr = ld.ttr(tokenized_sent) #lexical-diversity & type-token ratio
					scores.append(ttr)
					sents.append([title, sent])

		low_to_high_ld = [sent for _, sent in sorted(zip(scores, sents))]
		scores = sorted(scores)

		return low_to_high_ld, scores

class t_SNE:
	"""
	Class that performs t-SNE, a tool to visualize high-dimensional data.
	It is highly recommended to use another dimensionality reduction method 
	(e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the 
	number of dimensions to a reasonable amount (e.g. 50) if the number of 
	features is very high.
	Because it is so computationally costly, especially useful for smaller
	datasets (which is often the case in attribution).

	Parameters
	---------
	
	
	Returns
	-------
	plot visualizing t-SNE

	"""
	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot():
		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		invalid_words = []
		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='normalizer', #for some reason works better than standard-scaling
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		pca = PCA(n_components=50)
		pca_X = pca.fit_transform(X)
		tsne = TSNE(n_components=2, perplexity=25, learning_rate=0.001, n_iter=40000)
		tsne_X = tsne.fit_transform(pca_X)

		fig = plt.figure(figsize=(4.7,3.2))
		ax = fig.add_subplot(111)
		x1, x2  =  tsne_X[:,0], tsne_X[:,1]

		for index, (p1, p2, a, title) in enumerate(zip(x1, x2, authors, titles)):
			
			full_title = title.split('_')[0]
			sample_number = title.split('_')[-1]
			abbrev = title.split('_')[0].split('-')
			abbrev = '.'.join([w[:3] for w in abbrev]) + '-' + sample_number

			if a == 'Adso-Dervensis':
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.17, edgecolors ='k', linewidths = 0.2)
			elif a == 'Ioannis-Sancti-Arnulfi':
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.4, edgecolors ='k', linewidths = 0.2)
			else:
				ax.scatter(p1, p2, marker='o', color='k', \
					s=40, zorder=2, alpha = 0.8, edgecolors ='k', linewidths = 0.2)
				# ax.text(p1+0.5, p2+0.5, abbrev, color='black', fontdict={'size': 6}, zorder=1)

			ax.set_xlabel('t-SNE-1')
			ax.set_ylabel('t-SNE-2')

		plt.tight_layout()
		plt.show()
		
		fig.savefig("", \
					dpi=300, transparent=True, format='png')

class DBS:
	"""
	Class that performs DBScan
	Code inspired by https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
	"""
	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def plot():

		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		invalid_words = []
		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='normalizer', #for some reason works better than standard-scaling
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		pca = PCA(n_components=2)
		X = pca.fit_transform(X)

		# #############################################################################
		# Compute DBSCAN

		# epsilon range or 𝜖 indicates predefined distance around each point, (decisive whether core, border or noise point)
		eps = 0.46
		min_samples = len(X)

		db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
		labels = db.labels_

		no_clusters = len(np.unique(labels)) # number of clusters
		no_noise = np.sum(np.array(labels) == -1, axis=0) # number of clusters

		print('Estimated no. of clusters: %d' % no_clusters)
		print('Estimated no. of noise points: %d' % no_noise)

		# Plot results
		# Black removed and is used for noise instead.

		# Generate scatter plot for training data
		colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
		plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
		plt.xlabel('Axis X[0]')
		plt.ylabel('Axis X[1]')

		plt.show()

		fig.savefig("", \
					dpi=300, transparent=True, format='png')

class StyleChangeDetection:
	"""
	Class that detects intrinsic style change
	Intrinsic stylometric analysis of document, without referring to external documents or corpora for comparison
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.sample_len = sample_len

	def go():
		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='y', shingle_titles=['Vita-Deicoli'])

		invalid_words = []
		X, features, scaling_model = Vectorizer(texts, invalid_words,
												  n_feats=n_feats,
												  feat_scaling='standard_scaler',
												  analyzer='word',
												  vocab=list_of_function_words
												  ).tfidf(smoothing=False)

		for author, title, vec in zip(authors, titles, X):
			split_title = title.split('-')
			index_begin = int(split_title[-2])
			index_end = int(split_title[-1])

			distances = []
			for y, y_title, y_vec in zip(authors, titles, X):
				y_split_title = y_title.split('-')
				y_index_begin = int(y_split_title[-2])
				y_index_end = int(y_split_title[-1])

				x_indices = list(range(index_begin, index_end))
				y_indices = list(range(y_index_begin, y_index_end))
				overlapping_indices = [x for x in x_indices if x in y_indices]
				if len(overlapping_indices) == 0: # if there is no lexical overlap
					distance = euclidean_distances(vec.reshape(1, -1), y_vec.reshape(1, -1))
					distance = distance.ravel()[0]
					distances.append(distance)

			distances = np.array(distances)
			average_outlying = np.average(distances)

			print(title)
			print(average_outlying)
			print()

		plot_barplot 
		# index 

def plot_contours(ax, clf, xx, yy, **params):
	"""Plot the decision boundaries for a classifier.

	Parameters
	----------
	ax: matplotlib axes object
	clf: a classifier
	xx: meshgrid ndarray
	yy: meshgrid ndarray
	params: dictionary of params to pass to contourf, optional
	"""
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	out = ax.contourf(xx, yy, Z, **params)
	return out

def make_meshgrid(x, y, h=.02):
	"""Create a mesh of points to plot in

	Parameters
	----------
	x: data to base x-axis meshgrid on
	y: data to base y-axis meshgrid on
	h: stepsize for meshgrid, optional

	Returns
	-------
	xx, yy : ndarray
	"""
	x_min, x_max = x.min() - 1, x.max() + 1
	y_min, y_max = y.min() - 1, y.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	return xx, yy

class SVM_db_visualization:
	def __init__(self, folder_location):
		self.folder_location = folder_location
		self.model_location = model_location

	def go():
		
		sample_len = 500
		n_feats = 250

		function_words_only = open('/Users/...').read().split()
		invalid_words = ['comma', 'period', 'semicolon', 'exclamationmark', 'questionmark']

		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		"""
		MAKE 2D CLASSIFIER
		------------------
		"""
		# Load classifier parameters from the gridsearch into this 2D-simulation
		# Decision function = 'ovr' (one vs rest) or 'ovo' (one vs one)

		model = model_location
		model_name = model.split('/')[-1]
		grid = pickle.load(open(model, 'rb'))

		best_model = grid.best_estimator_

		selected_features = best_model['vectorizer'].get_feature_names()
		vectorizer = best_model.named_steps['vectorizer']
		vectorizer = vectorizer.set_params(vocabulary=selected_features)
		scaler = best_model.named_steps['feature_scaling']
		# dim_red = best_model.named_steps['reduce_dim']

		X_train = vectorizer.fit_transform(texts).toarray()
		# x_test = vectorizer.transform(test_texts).toarray()

		label_dict = {}
		inverse_label = {}
		for title in authors: 
			label_dict[title.split('_')[0]] = 0 
		for i, key in zip(range(len(label_dict)), label_dict.keys()):
			label_dict[key] = i
			inverse_label[i] = key

		Y_train = []
		for title in authors:
			label = label_dict[title.split('_')[0]]
			Y_train.append(label)
		Y_train = np.array(Y_train)

		"""
		SCALING
		-------
		"""
		# Retrieve all original scaling weights from best model
		# Apply selected scaling weights to selected features

		feature_scaling = {feat: (mean, scale) for mean, scale, feat in 
						   zip(scaler.mean_, \
							   scaler.scale_, \
							   selected_features)}
		model_means = []
		model_vars = []
		for feature in selected_features:
			model_means.append(feature_scaling[feature][0])
			model_vars.append(feature_scaling[feature][1])
		model_means = np.array(model_means)
		model_vars = np.array(model_vars)

		X_train = (X_train - model_means) / model_vars
		# x_test = (x_test - model_means) / model_vars

		"""
		PRINCIPAL COMPONENTS ANALYSIS
		-----------------------------
		"""
		# Visualize decision boundary with PCA
		# Unfortunately, we have to instantiate a new SVM model, one that
		# refits on data that has become 2dimensional.
		# I did not find a way to do have a decision hyperplane become 2d.
		# Transform with same PCA values

		pca = PCA(n_components=2)
		X_train = pca.fit_transform(X_train)
		# x_test = pca.transform(x_test)
		xx, yy = make_meshgrid(X_train[:,0], X_train[:,1])
		var_exp = pca.explained_variance_ratio_
		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)
		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(selected_features, loadings[:,0]), \
								  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(selected_features, loadings[:,1]), \
						   		  key=lambda tup: tup[1], reverse=True)

		"""
		RANK FEATURES
		-------------
		"""
		# Plot only the z features that on average have the highest weight
		# Choose number of features per principal component to be plotted (slice from vocab_weights)
		# Set threshold as to which 'best discriminators in the PC' may be plotted

		z = 50

		scaler = MinMaxScaler()
		scaled_loadings = scaler.fit_transform(np.abs(loadings))
		ranks = scaled_loadings.mean(axis=1)

		scaler = MinMaxScaler()
		ranks = scaler.fit_transform(ranks.reshape(-1,1))
		ranks = ranks.flatten()

		high_scorers = []
		font_dict_scores = []
		for idx, (feature, rank, coords) in enumerate(sorted(zip(selected_features, ranks, loadings), \
									key=lambda tup:tup[1], reverse=True)):
			print(rank)
			font_size = rank * 20
			font_dict_scores.append(font_size)
			if idx in list(range(0,z)):
				high_scorers.append(feature)

		# z = 5
		# printed_features_p1 = vocab_weights_p1[:z] + vocab_weights_p1[-z:]
		# printed_features_p2 = vocab_weights_p2[:z] + vocab_weights_p2[-z:]

		print("Explained variance: ", explained_variance)
		print("Number of words: ", n_feats)
		print("Sample size : ", sample_len)

		"""
		MAKE 2D CLASSIFIER
		------------------
		"""
		# Load classifier parameters from the gridsearch into this 2D-simulation
		# Decision function = 'ovr' (one vs rest) or 'ovo' (one vs one)
		best_clf = best_model.named_steps['classifier']
		clf_params = best_clf.get_params()
		svm_clf = svm.SVC(kernel=clf_params['kernel'], 
						  C=clf_params['C'], 
						  decision_function_shape=clf_params['decision_function_shape'])
		svm_clf.fit(X_train, Y_train)

		"""
		PLOTTING
		--------
		http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
		"""

		# FONTS
		# Custom fonts should be added to the matplotlibrc parameter file: cd '/Users/...'
		# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib
		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['ArnoPro-Regular']

		# Concatenation of training and test data
		# all_titles = train_titles + test_titles
		# all_props = train_props + test_props
		# all_coords = np.concatenate((X_train, x_test), axis=0)

		x1, x2 = X_train[:,0], X_train[:,1]
		# t1, t2 = x_test[:,0], x_test[:,1]

		# Define contour colours (light background)
		contour_dict = {'Goscelinus': '#C7DEE4',
						'Folcardus': '#A7CEA2'}

		# Define node colours
		colors_dict= {'Goscelinus': '#8BA5DF',
						'Folcardus': '#91CC83'}

		"""----------
		plot contours
		"""
		fig = plt.figure(figsize=(4.5,3))
		ax = fig.add_subplot(111)

		# Fixed list of hex color codes (custom cmap for contours) needs to be made
		# This equals making a custom cmap!
		# inspiration: https://stackoverflow.com/questions/9707676/defining
		# -a-discrete-colormap-for-imshow-in-matplotlib
		bounds = []
		listed_colors = []
		for title, label in label_dict.items():
			idx = label
			bounds.insert(idx, label)
			listed_colors.insert(idx, contour_dict[title])
		contour_cmap = colors.ListedColormap(listed_colors)
		norm = colors.BoundaryNorm(bounds, contour_cmap.N)

		plot_contours(ax, svm_clf, xx, yy, cmap=contour_cmap, alpha=0.8)
		
		"""---------------
		plot train samples
		"""
		for index, (p1, p2, author, title) in enumerate(zip(X_train[:, 0], X_train[:, 1], authors, titles)):
			ax.scatter(p1, p2, color=colors_dict[author], s=15, zorder=10, edgecolors='k', linewidth=0.3)

		"""----------
		plot loadings
		"""
		ax2 = ax.twinx().twiny()
		l1, l2 = loadings[:,0], loadings[:,1]
		for x, y, l, font_size in zip(l1, l2, selected_features, font_dict_scores):
			if l in high_scorers:
				ax2.text(x, y, l, ha='center', va="center", color='k', fontdict={'size': font_size}, zorder=2)

		# Important to adjust margins first when function words fall outside plot
		# This is due to the axes aligning (def align).
		# ax2.margins(x=0.2, y=0.2)

		# align_xaxis(ax, 2.4, ax2, 0)
		# align_yaxis(ax, 1.8, ax2, 0)

		align_xaxis(ax, 0, ax2, 0)
		align_yaxis(ax, 3, ax2, 0)

		plt.axhline(y=0, ls="--", lw=0.25, c='black', zorder=1)
		plt.axvline(x=0, ls="--", lw=0.25, c='black', zorder=1)

		"""-----------------------------
		plot layout and plotting command
		"""
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/gosc_folc_db_visualization.pdf", transparent=True, format='pdf')		

def add_one_by_one(l):
	new_l = []
	cumsum = 0
	for elt in l:
		cumsum += elt
		new_l.append(cumsum)
	return new_l

def syllabify_and_schematize(data, syllabifier):
	# input is data = [token, lemma, scanned_token]
	# output = [token, lemma, scanned_token, 'de-pro-mit', '-uu']

	token = data[0] # e.g. depromit
	lemma = data[1] # e.g. dēprōmō
	scanned = data[2] # e.g. dēprōmĭt

	# Filter unwanted artefacts in diacritics
	# ā̆|ē̆|ō̆|ī̆|ū̆ are problematic to capture with regex, due to the "combining breve" (UNICODE U+0306) - actually two unicode characters in one.
	# https://www.fileformat.info/info/charset/UTF-8/list.htm
	
	# convert the ā̆
	diacritic_a = re.compile(u'\u0101\u0306', re.UNICODE)
	lemma = re.sub(diacritic_a, 'â', lemma)
	scanned = re.sub(diacritic_a, 'â', scanned)

	# convert the ē̆
	diacritic_e = re.compile(u'\u0113\u0306', re.UNICODE)
	lemma = re.sub(diacritic_e, 'ê', lemma)
	scanned = re.sub(diacritic_e, 'ê', scanned)

	# convert the ī̆
	diacritic_i = re.compile(u'\u012B\u0306', re.UNICODE)
	lemma = re.sub(diacritic_i, 'î', lemma)
	scanned = re.sub(diacritic_i, 'î', scanned)

	# convert the ō̆
	diacritic_o = re.compile(u'\u014D\u0306', re.UNICODE)
	lemma = re.sub(diacritic_o, 'ô', lemma)
	scanned = re.sub(diacritic_o, 'ô', scanned)

	# convert the ū̆
	diacritic_u = re.compile(u'\u016B\u0306', re.UNICODE)
	lemma = re.sub(diacritic_u, 'û', lemma)
	scanned = re.sub(diacritic_u, 'û', scanned)

	# convert the ȳ̆
	diacritic_y = re.compile(u'\u0233\u0306', re.UNICODE)
	lemma = re.sub(diacritic_u, 'ŷ', lemma)
	scanned = re.sub(diacritic_u, 'ŷ', scanned)
	
	syllables = syllabifier.syllabify(token) # e.g. ['de', 'pro', 'mit']
	# Apply same syllabification to scanned token
	len_vals = [len(char) for char in syllables]
	indices = list(add_one_by_one(len_vals))
	indices = indices[:-1]
	indices.insert(0,0)
	scanned_syllabified = [scanned[i:j] for i,j in zip(indices, indices[1:]+[None])]

	long_pattern = re.compile(r'ā|ē|ī|ō|ū|ȳ') # macron
	short_pattern = re.compile(r'ă|ĕ|ĭ|ŏ|ŭ') # breve
	ambiguous_pattern = re.compile(r'â|ê|î|ô|û|ŷ') # These are ambiguous quantities, e.g. ā̆|ē̆|ō̆|ī̆|ū̆

	syllab_schema = []
	for syllab in scanned_syllabified:
		long_result = long_pattern.findall(syllab)
		short_result = short_pattern.findall(syllab)
		ambiguous_result = ambiguous_pattern.findall(syllab)
		if len(long_result) == 1:
			syllab_schema.append('—')
		elif len(short_result) == 1:
			syllab_schema.append('u')
		elif len(ambiguous_result) == 1:
			syllab_schema.append('x')

	scanned_syllabified = '-'.join(scanned_syllabified)
	syllab_schema = ''.join(syllab_schema)

	data.append(scanned_syllabified)
	data.append(syllab_schema)

	return data

def lemmatize_function(token):
	lemmatizer = LemmaReplacer('latin')
	lemma = lemmatizer.lemmatize(token)

	return lemma

def accentuate(data):
	if data[0] not in ['COMMA', 'PERIOD', 'COLON', 'SEMICOLON', 'EXCLAMATIONMARK', 'QUESTIONMARK']:
		scanned_syllabified = data[4]
		quantity_schema = data[5]
		syllables = scanned_syllabified.split('-')
		n_syllables = len(syllables)

		# accent either paroxytone, proparoxytone
		if n_syllables == 1:
			quality_schema = '1'
		elif n_syllables == 2:
			accent = 'p'
			quality_schema = str(n_syllables) + accent
		elif n_syllables >= 3:
			try:
				penult = quantity_schema[-2]
				if penult == '—': # if the penult is long
					accent = 'p'
					quality_schema = str(n_syllables) + accent
				elif penult == 'u': # if the penult is light / short
					accent = 'pp'
					quality_schema = str(n_syllables) + accent
				elif penult == 'x':
					accent = 'x'
					quality_schema = str(n_syllables) + accent
			except IndexError:
				quality_schema = 'unknown'

		data.insert(len(data), quality_schema)


def clean_lines(filename):
	# Input 
	# *** raw Collatinus output
	#
	# Output 
	# token gets syllabified here and a quantitative schema is devised (syllabify_and_schematize)
	# on basis of quantities, token receives qualitative information
	# *** requieuit	v1	rĕquĭēsco	rĕquĭēvĭt	rĕ-quĭ-ēv-ĭt	uu—u	4p
	# *** token 	postag 	lemma 	scansion 	syllabification 	quantity_schema 	quality_schema

	syllabifier = Syllabifier()	
	
	# .csv file gets 'cleaned' .txt equivalent
	new_filename = filename.replace(".csv", ".txt") 
	new_fob = open(new_filename, 'w')

	for line in open(filename):
		line = str(line)
		token = line.split()[3]
		postag = line.split()[4]
		pattern = re.compile(r'(?<=	)(\w+)[ă|ā|ĕ|ē|ĭ|ī|ō|ŏ|ū|ŭ|ā̆|ē̆|ō̆|ī̆|ū̆](\w+)')
		result = pattern.finditer(line)
		data = [each[0] for each in result]
		data.insert(0, token)
		try:
			data = syllabify_and_schematize(data, syllabifier)

		except IndexError:
			parsed_line = line.split('\t')
			try:
				lemma = parsed_line[6].split(',')[0]
				scanned = parsed_line[-1].split(' ')
				if scanned[1] == '+':
					scanned_token = ''.join([scanned[0], scanned[2]])
				else:
					scanned_token = scanned[0]
				data = [token, lemma, scanned_token]
				data = syllabify_and_schematize(data, syllabifier)

			except IndexError:
				if line.split()[4] == "unknown":
					if line.split()[3] in ['COMMA', 'PERIOD', 'COLON', 'SEMICOLON', 'EXCLAMATIONMARK', 'QUESTIONMARK']:
						data = [token]
					else:
						lemma = lemmatize_function(token)[0]
						scanned_syllabified = '-'.join(syllabifier.syllabify(token))
						data = [token, lemma, 'unknown', scanned_syllabified, ''.join(['x' for i in scanned_syllabified.split('-')])]
				else:
					data = [token, line.split()[6], line.split()[-1]]
					data = syllabify_and_schematize(data, syllabifier)

		data.insert(1, postag)
		accentuate(data)
			
		new_line = '\t'.join(data) + '\n'
		new_fob.write(new_line)

class CollatinusPreprocess:
	"""
	Class that structurally takes on raw Collatinus output,
	and serves merely to send those files through the def clean_files function above.
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location

	def go():
		print(folder_location)
		for author_folder in glob.glob(folder_location + '/*'):
			if author_folder.split('/')[-1] in ['Goscelinus']:
				for folder in glob.glob(author_folder + '/*'):
					for filename in glob.glob(folder + '/*'):
						clean_lines(filename)
			else:
				for filename in glob.glob(author_folder + '/*'):
					clean_lines(filename)

class FileSplitter:
	"""
	Class that takes files from cleaned_and_corrected looking like this:
	ipsius	p41	īpsĕ	īpsī̆ŭs	īp-sî-ŭs	—xu	3x
	token 	postag	 lemma 	scanned_token 	scanned_and_syllabified 	quantity 	quality

	it then distributes this information to separate folders.
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location

	def split():
		print(folder_location)
		for author_folder in glob.glob(folder_location + '/*'):
			if author_folder.split('/')[-1] in ['Goscelinus']:
				for folder in glob.glob(author_folder + '/*'):
					for filename in glob.glob(folder + '/*'):
						filename_lemmas = filename.replace('3_cleaned-and-corrected', '4_only-lemmas')
						filename_syllables = filename.replace('3_cleaned-and-corrected', '5_only-syllables')
						filename_quantity = filename.replace('3_cleaned-and-corrected', '6_only-quantity')
						filename_quality = filename.replace('3_cleaned-and-corrected', '7_only-quality')
						filename_postags = filename.replace('3_cleaned-and-corrected', '8_only-postags')

						fob_lemmas = open(filename_lemmas, 'w')
						fob_syllables = open(filename_syllables, 'w')
						fob_quantity = open(filename_quantity, 'w')
						fob_quality = open(filename_quality, 'w')
						fob_postags = open(filename_postags, 'w')
						
						for line in open(filename):
							line = line.split()
							try:
								postag = line[1]
								lemma = line[2]
								syllable = ' '.join(line[4].split('-'))
								quantity = line[5]
								quality = line[6]
							except IndexError: # filters out punctuation or mistakes
								postag = line[0]
								lemma = line[0]
								syllable = line[0]
								quantity = line[0]
								quality = line[0]

							fob_postags.write(postag + ' ')
							fob_lemmas.write(lemma + ' ')
							fob_syllables.write(syllable + ' ')
							fob_quantity.write(quantity + ' ')
							fob_quality.write(quality + ' ')
			else:
				for filename in glob.glob(author_folder + '/*'):
					filename_lemmas = filename.replace('3_cleaned-and-corrected', '4_only-lemmas')
					filename_syllables = filename.replace('3_cleaned-and-corrected', '5_only-syllables')
					filename_quantity = filename.replace('3_cleaned-and-corrected', '6_only-quantity')
					filename_quality = filename.replace('3_cleaned-and-corrected', '7_only-quality')
					filename_postags = filename.replace('3_cleaned-and-corrected', '8_only-postags')

					fob_lemmas = open(filename_lemmas, 'w')
					fob_syllables = open(filename_syllables, 'w')
					fob_quantity = open(filename_quantity, 'w')
					fob_quality = open(filename_quality, 'w')
					fob_postags = open(filename_postags, 'w')
					
					for line in open(filename):
						line = line.split()
						try:
							postag = line[1]
							lemma = line[2]
							syllable = ' '.join(line[4].split('-'))
							quantity = line[5]
							quality = line[6]
						except IndexError: # filters out punctuation or mistakes
							postag = line[0]
							lemma = line[0]
							syllable = line[0]
							quantity = line[0]
							quality = line[0]

						fob_postags.write(postag + ' ')
						fob_lemmas.write(lemma + ' ')
						fob_syllables.write(syllable + ' ')
						fob_quantity.write(quantity + ' ')
						fob_quality.write(quality + ' ')

def Sort_Tuple(tup):
	
	# getting length of list of tuples
	lst = len(tup) 
	for i in range(0, lst): 
		  
		for j in range(0, lst-i-1): 
			if (tup[j][1] > tup[j + 1][1]): 
				temp = tup[j] 
				tup[j]= tup[j + 1] 
				tup[j + 1]= temp 
	return tup

class FeatureSelection:
	"""
	Class that takes as input feature vectors and labels
	and identifies attributes likely dependent on class.
	"""

	def __init__(self, folder_location):
		self.folder_location = folder_location

	def go_chi2(self): # filter method for feature selection

		"""
		PARAMETERS
		----------
		# State parameters
		# Load in files and load vectorizer
		"""

		sample_len = 500
		n_feats = 300
		n_kbest = 41

		function_words_only = open('/Users/...').read().split()
		invalid_words = ['opus']

		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])

		"""
		ENCODING X_TRAIN, x_test AND Y_TRAIN, y_test
		--------------------------------------------
 		"""
		# Arranging dictionary where title is mapped to encoded label
		# Ultimately yields Y_train

		label_dict = {}
		inverse_label = {}
		for title in authors: 
			label_dict[title.split('_')[0]] = 0 
		for i, key in zip(range(len(label_dict)), label_dict.keys()):
			label_dict[key] = i
			inverse_label[i] = key

		Y_train = []
		for title in authors:
			label = label_dict[title.split('_')[0]]
			Y_train.append(label)

		y = np.array(Y_train)

		X, features, scaling_model = Vectorizer(texts, invalid_words,
									  n_feats=n_feats,
									  feat_scaling='standard_scaler',
									  analyzer='word',
									  vocab=function_words_only).raw()

		# Uncomment for visualizing rhymes on top of PCA plot
		replace_pattern  = re.compile('ϑ') # strange sign to mask '-'
		replace_pattern_2  = re.compile('ççç') # strange sign to mask ' : '
		replace_pattern_3 = re.compile('ς') # strange sign to mask '_'
		features = [re.sub(replace_pattern, '-', feat) for feat in features]
		features = [re.sub(replace_pattern_2, ' : ', feat) for feat in features]
		features = [re.sub(replace_pattern_3, '_', feat) for feat in features]

		positivized = [] # chi-square cannot be calculated on negative values
		for idx in range(0, X.shape[-1]):
			feat_freqs = X[:,idx] + np.abs(X[:,idx].min())
			positivized.append(feat_freqs)
		b = np.array(positivized)
		X = np.transpose(b)

		chi2_model = SelectKBest(chi2, k=n_kbest) # 'kbest selection' determines what shows in the plot as well
		chi2_model.fit(X, y)

		X_new = chi2_model.fit_transform(X, y) # transforms original X to k best features
		
		dict_results = {}
		results = []
		for feature, score, pvalue in zip(features, chi2_model.scores_, chi2_model.pvalues_):
			dict_results[feature] = (score, pvalue)
			results.append((feature, score, pvalue))
		chi2_ordered = Sort_Tuple(results)

		# nested_dict = {author A: {'et': [23, 24, 20, ...], 'in': [18, 16, 19, ...]}, author B: {}}
		nested_dict = {}
		
		for author in authors:
			nested_dict[author] = {}
		
		for author in authors:
			for feat in features:
				nested_dict[author][feat] = []

		for author, sample in zip(authors, texts):
			vectorizer = CountVectorizer(vocabulary=features)
			freqs = vectorizer.fit_transform([sample]).toarray()
			freqs = np.ravel(np.sum(freqs, axis=0))
			feats = vectorizer.get_feature_names()
			for feat, freq in zip(feats, freqs):
				nested_dict[author][feat].append(freq)

		# Initialize dictionary
		# dict = {'et': [rank for auth A, rank for auth B], 'in': [rank for auth A, rank for auth B]}
		feature_ranks = {}
		for feature in features: 
			feature_ranks[feature] = [0,0]

		for author, feat_freq_dict in nested_dict.items():
			listje = []
			for feat, freqs_list in feat_freq_dict.items():
				average_frequency = np.average(freqs_list)
				listje.append((feat, average_frequency))

			# Get author index so that there is no shifting around
			author_idx = label_dict[author]
			listje = Sort_Tuple(listje)
			sorted_feats = [i[0] for i in listje]
			sorted_average_freqs = [i[1] for i in listje]

			for idx, (feat, freq) in enumerate(zip(sorted_feats, sorted_average_freqs)):
				# print(len(features))
				# rank = (idx + 1) / len(features)
				rank = (idx + 1)
				feature_ranks[feat][author_idx] = rank
		
		# print(feature_ranks)
		kbest = np.asarray(features)[chi2_model.get_support()]

		chi2_scores = chi2_model.scores_
		chi2_pvalues = chi2_model.pvalues_

		"""
		PRINT RESULTS
		"""
		results_file = open('/Users/...')

		author_a_preferences = []
		author_b_preferences = []
		for feat, score, pvalue in zip(features, chi2_scores, chi2_pvalues):
			if feat in kbest: # n best features (chi2)
				x, y = feature_ranks[feat] # x = rank of author A | y = rank of author B
				if x > y: 
					author_a_preferences.append((feat, score, pvalue))
				else:
					author_b_preferences.append((feat, score, pvalue))

		author_a_preferences = Sort_Tuple(author_a_preferences)[::-1]
		author_b_preferences = Sort_Tuple(author_b_preferences)[::-1]

		for (feat, score, pvalue) in author_a_preferences:
			results_file.write('Goscelin\t' + str(feat)+ '\t' + str(score) +'\t' + str(pvalue) + '\n')
		for (feat, score, pvalue) in author_b_preferences:
			results_file.write('Vita Amelbergae' + str(feat)+ '\t' + str(score) +'\t' + str(pvalue) + '\n')

		"""
		VISUALIZE
		"""
		# the chi2 scores will determine the colour dict alpha value
		normalized_chi2_scores = (chi2_scores - np.min(chi2_scores))/np.ptp(chi2_scores) # normalize to [0,1] range
		normalized_chi2_pvalues = (chi2_pvalues - np.min(chi2_pvalues))/np.ptp(chi2_pvalues) # normalize to [0,1] range

		colour_dict = {'Wulfstan-of-Winchester': '#4A94BA',
					   'Byrhtferth-of-Ramsey': '#C4D200',
					   'Aethelwold-A': 'r',
					   'Folcardus': '#91CC83',
					   'Goscelinus': '#8BA5DF',
					   'Vita-et-Inventio-Amelbergae': '#FFAEAE'}

		ticks_dict = {'Vita-et-Inventio-Amelbergae': 'x', 'Goscelinus': 'x'}

		# rcParams['font.family'] = 'sans-serif'
		# rcParams['font.sans-serif'] = ['ArnoPro-Regular']
		rcParams['font.family'] = 'Times New Roman'

		fig = plt.figure(figsize=(4.7,3.2))

		ax2 = fig.add_subplot(111)

		ax2.set_xlabel('Goscelin')
		ax2.set_ylabel('Vita and Inventio Amelbergae')

		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)

		ax2.plot([0, 1], [0, 1], transform=ax2.transAxes, ls="--", c=".3", linewidth=0.5) # diagonal line

		for feat, score, pvalue in zip(features, normalized_chi2_scores, normalized_chi2_pvalues):
			x, y = feature_ranks[feat]

			if x > y: 
				colour = colour_dict['Goscelinus']
				tick = ticks_dict['Goscelinus']
				a = 0
				alpha = 1*(x/len(features))
			else:
				colour = colour_dict['Vita-et-Inventio-Amelbergae']
				tick = ticks_dict['Vita-et-Inventio-Amelbergae']
				a = 1
				alpha = 1*(y/len(features))
			
			ax2.scatter(x, y, marker=tick, color='k', s=14, zorder=3, alpha=alpha, linewidth=0.3)

			if feat in kbest: # n best features (chi2)
			# if feat in ['haud']:
				if a == 0:
					ax2.text(x, y-0.1, feat, color='k', fontdict={'size': 5*(1+score)})
					l = mlines.Line2D([x, x], [y, y-0.1], linewidth=0.1, color=colour, alpha=alpha)
					ax2.add_line(l)
				else:
					ax2.text(x-0.1, y, feat, color='k', fontdict={'size': 5*(1+score)})
					l = mlines.Line2D([x, x-0.1], [y, y], linewidth=0.1, color=colour, alpha=alpha)
					ax2.add_line(l)

		# Set ticks of indices
		xticks = [0, 100, 200, 300]
		xlabels = ['300', '200', '100', '1']
		yticks = [0, 100, 200, 300]
		ylabels = ['300', '200', '100', '1']

		ax2.tick_params(axis='both', which='major', labelsize=8)
		plt.xticks(xticks, xlabels, rotation='horizontal')
		plt.yticks(yticks, ylabels, rotation='horizontal')

		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/chi-2-no-colour-vAmel.png", dpi=300, transparent=True, format='png', bbox_inches='tight')

class RollingAnalysis:
	def __init__(self, test_folder_location, model_location):
		self.test_folder_location = test_folder_location
		self.model_location = model_location

	def roll(test_folder_location, model_location):
		authors, titles, texts = DataReader(test_folder_location, sample_len).fit(shingling='y', shingle_titles=['Vita-Ædwardi'])

		# Load classifier and refit test data to its specific parameters
		model = model_location
		model_name = model.split('/')[-1]
		grid = pickle.load(open(model, 'rb'))

		best_model = grid.best_estimator_

		best_clf = best_model.named_steps['classifier']
		selected_features = best_model['vectorizer'].get_feature_names()
		vectorizer = best_model.named_steps['vectorizer']
		vectorizer = vectorizer.set_params(vocabulary=selected_features)
		scaler = best_model.named_steps['feature_scaling']

		x_test = vectorizer.transform(texts).toarray()

		feature_scaling = {feat: (mean, scale) for mean, scale, feat in 
						   zip(scaler.mean_, \
							   scaler.scale_, \
							   selected_features)}
		model_means = []
		model_vars = []
		for feature in selected_features:
			model_means.append(feature_scaling[feature][0])
			model_vars.append(feature_scaling[feature][1])
		model_means = np.array(model_means)
		model_vars = np.array(model_vars)

		x_test = (x_test - model_means) / model_vars
		y_predict = best_clf.predict(x_test)
		y_probabilities = best_clf.predict_proba(x_test)

		for title, probabs in zip(titles, y_probabilities):
			print(title, probabs)

		"""
		VISUALIZATION
		"""

		# Collect window ranges for plotting on figure
		y = []
		for title in titles:
			title = title.split('-')
			begin, end = title[-2], title[-1]
			begin = np.int(begin)
			end = np.int(end)
			# y_spot = (begin + end) / 2
			y.append(end)
		y = np.array(y)

		# This is just an adjustment for the line plot.
		# Dots should center.
		mid_ys = [i-50 for i in y]

		fig = plt.figure(figsize=(10,3))
		# fig = plt.figure(figsize=(3.5,2))
		ax = fig.add_subplot(111)

		colour_dict = {'Goscelinus': '#8BA5DF',
						'Folcardus': '#91CC83'}

		# Plot impostors results
		for idx, (loc, prediction, probas) in enumerate(zip(y, y_predict, y_probabilities)):
			if idx == 0:
				width = -500
			else:
				width = -10
			if prediction == 0: 
				ax.bar(x=loc, height=probas[prediction], align='edge', width=width, color=colour_dict['Goscelinus'], alpha = probas[prediction])
			else:
				ax.bar(x=loc, height=probas[prediction], align='edge', width=width, color=colour_dict['Folcardus'], alpha = probas[prediction])

		# Layout elements

		# FONTS
		# Custom fonts should be added to the matplotlibrc parameter file: cd '/Users/...'
		# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib
		rcParams['font.family'] = 'sans-serif'
		rcParams['font.sans-serif'] = ['ArnoPro-Regular']

		ax.set_xlabel('Progression of the Life of King Edward the Confessor')
		ax.set_ylabel('Probability')
		# # Despine
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_visible(True)

		# Set ticks of indices
		xticks = [1, 1255, 2463, 3185, 4382, 5085, 5733, 6721]
		xlabels = ['i.1', 'i.3', 'i.4', 'i.5', 'i.6', 'Extra muros [...]', 'i.7', 'ii.1–3 + ii.11']

		ax.tick_params(axis='both', which='major', labelsize=8)
		plt.xticks(xticks, xlabels, rotation='horizontal')
		# plt.yticks(yticks, ylabels, zorder=0)
		# ax.set_xlabel(xlabels, rotation=0, fontsize=20, labelpad=20)
		ax.set_xlim(xmin=0, xmax=7273)
		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/rolling-analysis-new.pdf", transparent=True, format='pdf')

class CollateTexts():
	# Use of CollateX for collating text witnesses. 
	# Below some functions to analyze word differences.
	def __init__(self, folder_location):
		self.folder_location = folder_location # folder should contain two texts files that are collated

	def go(folder_location):
		
		file_csv_collation = open('/Users/...')

		collation = Collation() # instantiate model

		witnesses = []
		titles = []
		for filename in glob.glob(folder_location + '/*'):
			title = filename.split('/')[-1].split('.')[0]
			titles.append(title)
			witness = open(filename).read()
			witnesses.append(witness)
		
		collation.add_plain_witness(titles[0], witnesses[0])
		collation.add_plain_witness(titles[1], witnesses[1])
		alignment_table = collate(collation, layout='vertical', output="csv")
		file_csv_collation.write(alignment_table)

class CollationAnalysis():
	# feed alignment table
	# gives most common differences between variants
	def __init__(self, directory):
		self.directory = directory # folder should contain two texts files that are collated

	def go(directory):
		with open(directory) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			witnesses = []
			for witness in csv_reader:
				witnesses.append(witness)

			all_elements = []
			for (element_one, element_two) in zip(witnesses[0],witnesses[1]):
				if element_one != element_two:
					all_elements.append((element_one, element_two))
			
			diffs_counts = Counter(all_elements)
			print(diffs_counts.most_common(100))

class LinearRegressionClass():

	# X = feature vectors
	# y = ascending labels that correspond to decade, i.e. [1060, 1070, 1080, 1090, 1100]
	# time is linear, so linear regression attempts to see if there is a linear trend in the lexicon of Goscelin's oeuvre

	def __init__(self, train_directory):
		self.train_directory = train_directory # folder should contain two texts files that are collated

	def go(train_directory):

		raw_dating = {'Vita-Ædwardi': 0,
				'In-natale-S-Edwoldi': 0,
				'Inventio-Amelbergae': 0,
				'Vita-et-Inventio-Amelbergae': 0,
				'Vita-Kenelmi': 0,
				'Vita-Kenelmi-brevior': 0,
				'Passio-Eadwardi-regis-et-martyris': 1,
				'Vita-Edithae-no-verse': 1,
				'Vita-Wlsini(amended-Love)': 1,
				'Inventio-Yvonis[PL+Macray]': 3,
				'Lectiones-de-Hildelitha': 3,
				'Lectiones-Eormenhilde': 3,
				'Lectiones-Sexburge': 3,
				'Liber-confortatorius': 2,
				'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 3,
				'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 3,
				'Translatio-Wulfildae': 3,
				'Translatio-Yvonis': 3,
				'Visio-Alvivae': 3,
				'Vita-et-Miracula-Æthelburge': 3,
				'Vita-Vulfilda': 3,
				'Vita-Werburge': 3,
				'Historia-minor-Augustini': 4,
				'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 4,
				'Miracula-Augustini-maior': 4,
				'Translatio-Augustini-et-aliorum-sanctorum': 4,
				'Translatio-Mildrethe': 4,
				'Vita-Augustini': 4,
				'Vita-de-Iusto-archiepiscopo': 4,
				'Vita-Letardi': 4,
				'Vita-Mildrethe': 4,
				'Miracula-Aetheldrethe': 5,
				'Miracula-Wihtburge': 5,
				'Vita-Aetheldrethe-recensio-C': 5,
				'Vita-Aetheldrethe-recensio-D': 5,
				'Vita-Milburge': 5,
				'Vita-Sexburge': 5,
				'Vita-Wihtburge': 5}

		invalid_words = []
		function_words_only = open('/Users/...').read().split()
		
		# authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
		# X, features, scaling_model = Vectorizer(texts, invalid_words,
		# 							  n_feats=n_feats,
		# 							  feat_scaling='standard_scaler',
		# 							  analyzer='word',
		# 							  vocab=function_words_only).raw()
		# # text samples get labelled by phase in Goscelin's career [1, 2, 3, 4]
		# y = np.array([raw_dating[title.split('_')[0]] for title in titles])

		# test_title = ['Vita-et-Inventio-Amelbergae'] # indicate the test case

		# # X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=1)

		# X_test = []
		# y_test = []
		# X_train = []
		# y_train = []
		# for x_samp, y_label, title in zip(X, y, titles):
		# 	full_title = title.split('_')[0]
		# 	first_title_bit = title.split('_')[0]
		# 	if first_title_bit == 'Vita-et-Inventio-Amelbergae':
		# 		X_test.append(x_samp)
		# 		y_test.append(y_label)
		# 	else: 
		# 		X_train.append(x_samp)
		# 		y_train.append(y_label)
		# X_test = np.array(X_test)
		# y_test = np.array(y_test)
		# X_train = np.array(X_train)
		# y_train = np.array(y_train)

		"""
		TRAIN MODEL
		"""

		# sample_lens = [500, 1000, 2000, 3000] # length of sample / segment
		# n_feats = [50, 100, 200, 300, 400, 500]
		
		# all_scores = []
		# model_properties = []
		# for sample_len in sample_lens:
		# 	for n_feat in n_feats: 
		# 		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
		# 		X, features, scaling_model = Vectorizer(texts, invalid_words,
		# 									  n_feats=n_feat,
		# 									  feat_scaling='standard_scaler',
		# 									  analyzer='word',
		# 									  vocab=function_words_only).raw()
		# 		# text samples get labelled by phase in Goscelin's career [1, 2, 3, 4]
		# 		y = np.array([raw_dating[title.split('_')[0]] for title in titles])

		# 		test_title = ['Vita-et-Inventio-Amelbergae'] # indicate the test case

		# 		X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=1)

		# 		"""
		# 		LINEAR REGRESSION CLASSIFIER
		# 		"""
		# 		model = LinearRegression()
		# 		model.fit(X_train, y_train)
		# 		# cross-validate
		# 		scores = cross_val_score(model, X_train, y_train, cv=5)

		# 		y_pred = model.predict(X_dev)
		# 		score = metrics.accuracy_score(y_dev, np.round(y_pred))
		# 		all_scores.append(score)
		# 		param_one = str(sample_len)
		# 		param_two = str(n_feat)
		# 		model_properties.append(param_one + ' ' + param_two)

		# rcParams['font.family'] = 'sans-serif'
		# rcParams['font.sans-serif'] = ['ArnoPro-Regular']
		# rcParams['font.family'] = 'Times New Roman'

		# fig = plt.figure(figsize=(4.7,3.2))
		# ax2 = fig.add_subplot(111)

		# for idx, score in enumerate(all_scores):
		# 	ax2.scatter(idx, score, marker='o', color='w', s=10, edgecolors='k', linewidth=0.5, alpha=1)

		# ax2.set_xlabel('Model parameters')
		# ax2.set_ylabel('Accuracy')

		# xlabels = model_properties
		# xticks = [idx for idx, i in enumerate(model_properties)]
		# plt.xticks(xticks, xlabels, zorder=0, rotation='vertical')

		# ax2.spines['top'].set_visible(False)
		# ax2.spines['right'].set_visible(False)

		# plt.tight_layout()
		# plt.show()
		# fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/linear-regress-accuracies.pdf", bbox_inches='tight', transparent=True, format='pdf')

		"""
		LINEAR REGRESSION CLASSIFIER
		"""
		# model = LinearRegression()
		# model.fit(X_train, y_train)
		# # cross-validate
		# scores = cross_val_score(model, X_train, y_train, cv=5)

		# # y_pred = model.predict(X_test)

		# r2_score = model.score(X, y) # The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
		# coefs = model.coef_ # coefficients (w) to minimize residual sum of squaress
		# intercept = model.intercept_
		# high_chi2_words = ['denique','atque','eam ','quidem','ob','autem','forte','eo','deinde','tui']

		# feats_with_coefs = []
		# for coef, feat in zip(coefs, features):
		# 	absolute_value = np.abs(coef)
		# 	info = (feat, coef, absolute_value)
		# 	feats_with_coefs.append(info)
		# sorted_on_timeline = sorted(feats_with_coefs, key=lambda t: t[1], reverse=False)
		# order_dict = {tup[0]: idx+1 for idx, tup in enumerate(sorted_on_timeline)}
		# feats_with_coefs = sorted(feats_with_coefs, key=lambda t: t[2], reverse=True)

		# print(feats_with_coefs)

		# print('score: ', r2_score)
		# print('positive regression: ', features[np.argmax(coefs)]) # the highest coefs = positive regression
		# print('negative regression: ', features[np.argmin(coefs)]) # the lowest coefs = negative regression
		# print('intercept: ', intercept)
		# print('number of seen features: ', model.n_features_in_)

		"""
		VISUALIZE
		"""
		# rcParams['font.family'] = 'sans-serif'
		# rcParams['font.sans-serif'] = ['ArnoPro-Regular']
		# rcParams['font.family'] = 'Times New Roman'

		# fig = plt.figure(figsize=(4.7,3.2))
		# ax2 = fig.add_subplot(111)

		# # rank in order of importance
		# alpha_dict = {}
		# coefs_temp = np.abs([i[1] for i in feats_with_coefs])
		# for_colours = (coefs_temp - np.min(coefs_temp))/np.ptp(coefs_temp) # normalize to [0,1] range
		# alpha_dict = {tup[0]: alpha for tup, alpha in zip(feats_with_coefs, for_colours)}

		# for (feat, coef, absolute_coef) in feats_with_coefs:
		# 	order = order_dict[feat]
		# 	ax2.scatter(order, np.abs(coef), marker='o', color='w', s=10, edgecolors='k', linewidth=0.5, alpha=0)
		# 	ax2.text(order, np.abs(coef), feat, color='k', alpha=alpha_dict[feat], zorder=10001, fontdict={'size': 12*alpha_dict[feat]})

		# ax2.set_xlabel('Ranked features')
		# ax2.set_ylabel('Absolute coefficient value')

		# # Draw arrows (as black triangles: ">k"/"^k") at the end of the axes.  In each
		# # case, one of the coordinates (0) is a data coordinate (i.e., y = 0 or x = 0,
		# # respectively) and the other one (1) is an axes coordinate (i.e., at the very
		# # right/top of the axes).  Also, disable clipping (clip_on=False) as the marker
		# # actually spills out of the axes.
		# ax2.plot(1, 0, ">k", transform=ax2.get_yaxis_transform(), clip_on=False)
		# ax2.plot(0, 1, "^k", transform=ax2.get_xaxis_transform(), clip_on=False)

		# ax2.spines['top'].set_visible(False)
		# ax2.spines['right'].set_visible(False)

		# xticks = []
		# xlabels = []
		# plt.xticks(xticks, xlabels, rotation='horizontal')

		# plt.tight_layout()
		# plt.show()
		# fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/linear-regress.pdf", bbox_inches='tight', transparent=True, format='pdf')

		"""
		VISUALIZE THE GOOD PREDICTORS
		"""
		# ecce and cum

		authors, titles, texts = DataReader(folder_location, sample_len).fit(shingling='n', shingle_titles=[])
		X, features, scaling_model = Vectorizer(texts, invalid_words,
									  n_feats=n_feats,
									  feat_scaling=False,
									  analyzer='word',
									  vocab=function_words_only).raw()
		# text samples get labelled by phase in Goscelin's career [1, 2, 3, 4]
		y = np.array([raw_dating[title.split('_')[0]] for title in titles])

		# create an X containing only the freqs of a feature with a good positive / negative coefficient
		for column, feature in enumerate(features): 
			if feature in ['itaque']: # cum
				feature_idx = column

		one_feature_X = X[:,feature_idx].reshape(-1, 1) # gets column of only 'itaque'

		# plot this
		rcParams['font.family'] = 'Times New Roman'

		fig = plt.figure(figsize=(3.5,2))
		ax2 = fig.add_subplot(111)
		
		data_dict = {}
		for ylabel, freq in zip(y, one_feature_X):
			data_dict[ylabel] = []
		for ylabel, freq in zip(y, one_feature_X):
			data_dict[ylabel].append(freq)

		data_dict = OrderedDict(sorted(data_dict.items()))

		x_coords = []
		y_coords = []
		sorted_y = []
		for ylabel, freqs in data_dict.items():
			for i in range(0, len(freqs)):
				x_coords.append(ylabel)
				sorted_y.append(ylabel)
			y_coords.append(freqs)
		y_coords = np.array(sum(y_coords, []))
		x_coords = [idx for idx, y_label in enumerate(x_coords)]

		model = LinearRegression()
		model.fit(y_coords, x_coords)
		coefs = model.coef_ # coefficients (w) to minimize residual sum of squaress
		y_pred = model.predict(y_coords)

		yticks = [1, 2, 3, 4, 5, 6, 7, 8]
		ylabels = ['1', '2', '3', '4', '5', '6', '7', '8']
		plt.yticks(yticks, ylabels, zorder=0)
		# ax.set_xlabel(xlabels, rotation=0, fontsize=20, labelpad=20)

		ax2.set_xlabel('Text samples')
		ax2.set_ylabel('Frequency')

		ax2.plot(y_pred, y_coords, color='k', linestyle='--', linewidth=0.5) # plot regression line
		# print(len(x_coords))
		
		marker_dict = {1: '^', 2: '2', 3: '*', 4: 'x'}
		era_dict = {1: '1', 2: '2', 3: '3', 4: '4'}
		colour_dict= {1: 'r', 2: 'g', 3: 'b', 4: 'c'}
		for x_coord, y_coord, y_truelabel in zip(x_coords, y_coords, sorted_y):
			# ax2.scatter(x_coord, y_coord, marker='o', color='w', s=10, edgecolors='k', linewidth=0.5, zorder=1000)
			# ax2.scatter(x_coord, y_coord, marker=marker_dict[y_truelabel], color='k', s=10, alpha=1, zorder=1)
			ax2.scatter(x_coord, y_coord, marker='o', s=4, color='w', edgecolors='k', linewidth=0.2)
			ax2.text(x_coord, y_coord, era_dict[y_truelabel], fontdict={'size': 7})

		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)

		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/linear-regress-one-feat-2.png", dpi=300, transparent=True, format='png', bbox_inches='tight')

def make_token_rich(token):
	# accepts list of syllables and turns them into 'rich' syllables (so as to save information on syllable's position within token)
	rich_token = []
	max_idx = len(token) - 1
	if len(token) == 1: # monosyllabic words do not need hyphens
		rich_token.append(token[0])
	else:
		for idx, syllab in enumerate(token):
			if idx == 0:
				rich_syllab = syllab + 'ϑ'
			elif idx == max_idx:
				rich_syllab = 'ϑ' + syllab
			else:
				rich_syllab = 'ϑ' + syllab + 'ϑ'
			rich_token.append(rich_syllab)
	return rich_token

class RhymeDetector():
	def __init__(self, filename):
		self.filename = filename

	def go(filename):

		author = filename.split("/")[-1].split(".")[0].split("_")[0]
		title = filename.split("/")[-1].split(".")[0].split("_")[1]

		# Uncomment to write new file
		# new_filename = '/Users/...'.format(author, title)
		new_filename = '/Users/...'.format(author, title)
		new_fob = open(new_filename, 'w')

		fricatives = ['v', 'f']
		plosives = ['b', 'p']
		dentals = ['t', 'd']
		consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x']
		vowels = ['a', 'e', 'i', 'o', 'u', 'y']
		macrons = ['ā', 'ē', 'ī', 'ō', 'ū', 'ȳ']
		breves = ['ă', 'ĕ', 'ĭ', 'ŏ', 'ŭ']
		ambiguous_quantities = ['â', 'ê', 'î', 'ô', 'û', 'ŷ']
		special_vowels = {'ŷ': 'y', 'ȳ': 'y', 'â': 'a', 'ă': 'a', 'ā': 'a', 'ê': 'e', 'ĕ': 'e', 'ē': 'e', 'î': 'i', 'ĭ': 'i', 'ī': 'i',  'ô': 'o', 'ŏ': 'o', 'ō': 'o', 'û': 'u', 'ŭ': 'u', 'ū': 'u'}
		all_vowels = vowels + macrons + breves + ambiguous_quantities

		text = []
		for line in open(filename):
			# unpack tagged files structured as such per line:
			# token \t pos \t scanned_lemma \t scanned_token \t syllab_token \t quantity_scheme \t quality_scheme
			# break off iteration at occurrence of PERIOD, QUESTIONMARK, EXCLAMATIONMARK
			unpacked_line = line.split()
			if unpacked_line[0] in ['PERIOD', 'QUESTIONMARK', 'EXCLAMATIONMARK']:
				text.append('πρ') # period
			elif unpacked_line[0] in ['COMMA', 'COLON', 'SERMICOLON']:
				text.append('κμ') # comma
			else:
				try:
					text.append(unpacked_line[4])
				except IndexError: # vooral voor 'COMMA	unknown'
					pass
		
		# block of code that keeps track of the index the word has within total text
		# necessary for correct segmentation and localizing rhymes within the linear course of text
		# e.g. ['0 1 2 3 4 5 πρ', ' 7 8 9 10 11 12 πρ', ...]
		text_indices = []
		for (word_idx, word) in enumerate(text):
			if word == 'πρ':
				text_indices.append(word)
			else:
				text_indices.append(str(word_idx))
		text_indices = ' '.join(text_indices)
		text_indices = text_indices.split('πρ')

		punc_idcs = [] # keep track of global indices of punctuation
		for each in text_indices:
			try:
				last_idx = int(each.split()[-1]) + 1
			except IndexError:
				pass
			punc_idcs.append(str(last_idx))

		text_indices = [sent + punc_idx for sent, punc_idx in zip(text_indices, punc_idcs)] # re-adds separator 'πρ' (period)

		text = ' '.join(text)
		text = text.split('πρ') # ['prī-mŭs ă-gĭt quēs-tŭs ĕt con-so-la-mi-na tho-mus ', ' Pel-la cŭm de-mo-ni-bus mŏ-v-ĕt ēv-īn-cĭt- sĕ-cūn-dŭs ', ' tēr-tĭ-ŭs ig-ni-tis pēl-lĭt fās-tī-dĭ-ă v-ō-tīs ', ' ē-dīc-tīs sūm-ptīs quār-tŭs pĕ-tĭt ās-tră quâ-drī-gīs ']
		text = [sent + 'πρ' for sent in text] # re-adds separator 'πρ' (period); important for detecting homoioteleuton at sense pauses

		for x, (nth_sent, sent) in zip(text_indices, enumerate(text)):
			sent = sent.lower() # lowercase sentence (important for detecting alliteration)
			sent = sent.split(' ') # ['prī-mŭs', 'ă-gĭt', 'quēs-tŭs', 'ĕt', 'con-so-la-mi-na', 'tho-mus', '']
			sent_length = len(sent)
			sent = [token.split('-') for token in sent if token != ''] # [['prī', 'mŭs'], ['ă', 'gĭt'], ...]]

			# Indexation of ordinal position of tokens within sentence
			global_indices = [i for i in x.split(' ') if i != ''] # list of ordinal positions of tokens within global text
			sent_indices = [idx for idx, token in enumerate(sent)] # list of ordinal positions of tokens within sentence

			new_sent = []
			for token, global_index, sentence_index in zip(sent, global_indices, sent_indices):
				token.insert(0, sentence_index) # adds token's sentence index, e.g. [2, 'tēr', 'tĭ', 'ŭs']
				token.insert(0, int(global_index)) # adds token's global index within total text, e.g. [345, 'tēr', 'tĭ', 'ŭs']
				new_sent.append(token)
			sent = new_sent

			results_dict = {} # {center token: [(target_token, match, type, at_sense_pause)]}
			tokens = [''.join(token[2:]) for token in sent]
			for token in tokens:
				results_dict[token] = []

			for token_pair in combinations(sent, 2): 
				# itertools.combinations avoids duplicates (e.g. gaudium : dominum and dominum : gaudium)
				# and token never gets compared to itself
				token_pair = [*token_pair]
				# Make callable strings of center and target token 
				center_token = ''.join(token_pair[0][2:])
				target_token = ''.join(token_pair[1][2:])

				# filter out any combinations containing function words - can be uncommented
				function_words_only = open('/Users/...').read().split()
				ct_no_diacrs = [] # center token without diacritics
				tt_no_diacrs = [] # target token without diacritics
				for let in center_token:
					if let in special_vowels.keys():
						let = special_vowels[let]
					ct_no_diacrs.append(let)
				for let in target_token:
					if let in special_vowels.keys():
						let = special_vowels[let]
					tt_no_diacrs.append(let)
				ct_no_diacrs = ''.join(ct_no_diacrs)
				tt_no_diacrs = ''.join(tt_no_diacrs)
				if ct_no_diacrs in function_words_only:
					continue
				if tt_no_diacrs in function_words_only:
					continue

				at_sense_pause = False # do both items precede sense pauses, i.e. either 'πρ' (period) or 'κμ' (comma) - yes or no?
				sent_idx_center_token = token_pair[0][1] # [76, 0, 'prī', 'mŭs'], [77, 1, 'ă', 'gĭt']
				sent_idx_target_token = token_pair[1][1] # [76, 0, 'prī', 'mŭs'], [77, 1, 'ă', 'gĭt']
				global_idx_center_token = token_pair[0][0] # [76, 0, 'prī', 'mŭs'], [77, 1, 'ă', 'gĭt']

				try: 
					center_token_nbr = sent[sent_idx_center_token+1] # center token neighboring token
					target_token_nbr = sent[sent_idx_target_token+1] # target token neighboring token
					if center_token_nbr[-1] in ['πρ', 'κμ']:
						if target_token_nbr[-1] in ['πρ', 'κμ']:
							at_sense_pause += True
				except IndexError:
					pass

				# Make token rich (i.e. containing positional information of syllables)
				# This calls a function - see above
				# returns = ['cō-', '-gnō-', '-vĭ-', '-mŭs'] i.e. ['cōϑ', 'ϑgnōϑ', 'ϑvĭϑ', 'ϑmŭs']
				# Especially helpful once the matches need to be translated to readable style features
				for idx, each_token in enumerate(token_pair):
					if idx == 0: # make center token rich
						rich_center_token = make_token_rich(each_token[2:]) # only from idx 2 onward the word begins
						rich_center_token.insert(0, each_token[1]) # re-insert sentence index
						rich_center_token.insert(0, each_token[0]) # re-insert global index
					else: # make target token rich
						rich_target_token = make_token_rich(each_token[2:]) # only from idx 2 onward the word begins
						rich_target_token.insert(0, each_token[1]) # re-insert sentence index
						rich_target_token.insert(0, each_token[0]) # re-insert global index
				rich_token_pair = [rich_center_token, rich_target_token]
				
				# list will contain ultimae syllabae of both compared tokens
				ultim_syllabs = []
				rich_ultim_syllabs = [] 
				 # list will contain penultimae syllabae of both compared tokens
				penultim_syllabs = []
				rich_penultim_syllabs = [] 
				 # list will contain primae syllabae of both compared tokens
				prim_syllabs = []
				rich_prim_syllabs = [] 
				 # list will contain sentence idxs of compared tokens
				token_pair_sent_idxs = []
				
				# Fill up declared empty containers above with necessary token information
				for token, rich_token in zip(token_pair, rich_token_pair):
					token_pair_sent_idxs.append(token[1]) # sentence index info

					prim_syllabs.append(token[2]) # prima syllaba - in first position 
					rich_prim_syllabs.append(rich_token[2]) # prima syllaba - in first position 

					ultim_syllabs.append(token[-1]) # ultima syllaba
					rich_ultim_syllabs.append(rich_token[-1]) # ultima syllaba

					try: # penultima syllaba — can vary if there is a penultimate syllab...
						if isinstance(token[-2], int) == True: # index of token within same tup mistakenly became penultima syllaba sometimes
							penultim_syllabs.append('-')
							rich_penultim_syllabs.append('-')
						else:
							penultim_syllabs.append(token[-2]) # penultima syllaba
							rich_penultim_syllabs.append(rich_token[-2]) # penultima syllaba
					except IndexError:
						penultim_syllabs.append('-')
						rich_penultim_syllabs.append('-')

				# =======================================
				# STEP 1 - Absolute distance between center and target token for proximity score
				np_token_pair_idxs = np.array(token_pair_sent_idxs)
				absolute_dist = np.abs(np.diff(np_token_pair_idxs))[0] # abs distance between compared tokens in sentence

				# in a sentence of 2 words, you only have 1 shot of finding a rhyme - so if you find one, it should be boosted
				# in a sentence of 24 words, you have 23 chances - so if you do find one, it is less significant.
				rhyme_probability = 1/(sent_length-1)
				
				# =======================================
				# STEP 2 - detecting rhyme in ultima syllaba
				# function returns the matches between two final syllables of compared words
				# as such: [-1, -2, -3] (corresponding to last, penultimate, antepenult letter)
				ultim_syllabs_corrs = syllab_matcher(ultim_syllabs)

				# ******************************
				# match on idcs -2 alone if vowel (assonance)
				# match on idx -3 alone if vowel (assonance)
				# match on idx -1 alone (assonance or consonance)

				if ultim_syllabs_corrs in [[-2]]: # matching vowel on penult letter of ultim syll
					overlapping_letters = [syllab[-2] for syllab in ultim_syllabs]
					if overlapping_letters[0] in all_vowels:
						rhymetype = len(set(overlapping_letters))
						if rhymetype == 1:
							# assonance. i.e. '-ĭ-':'-ĭ-'
							alphabetized_match = sorted(['ϑ' + syllab[-2] + 'ϑ' for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'asϑpς'
							# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
						elif rhymetype == 2:
							# near assonance. i.e. '-ĭ-':'-ī-'
							alphabetized_match = sorted(['ϑ' + syllab[-2] + 'ϑ' for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'nearϑasϑpς'
							# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
				
				elif ultim_syllabs_corrs in [[-3]]: # matching vowel on antepenult letter of ultim syll
					overlapping_letters = [syllab[-3] for syllab in ultim_syllabs]
					if overlapping_letters[0] in all_vowels:
						rhymetype = len(set(overlapping_letters))
						if rhymetype == 1:
							# assonance. i.e. '-ĭ-':'-ĭ-'
							alphabetized_match = sorted(['ϑ' + syllab[-3] + 'ϑ' for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'asϑapς'
							# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
						elif rhymetype == 2: 
							# near assonance. i.e. '-ĭ-':'-ī-'
							alphabetized_match = sorted(['ϑ' + syllab[-3] + 'ϑ' for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'nearϑasϑapς'
							# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
				
				elif ultim_syllabs_corrs in [[-1]]: # matching vowel or consonant on last letter of ultim syll
					overlapping_letters = [syllab[-1] for syllab in ultim_syllabs]
					if overlapping_letters[0] in all_vowels:
						rhymetype = len(set(overlapping_letters))
						if rhymetype == 1:
							# assonance. i.e. '-ĭ':'-ĭ'
							alphabetized_match = sorted(['ϑ' + syllab[-1] for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'asϑuς'
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
						elif rhymetype == 2: 
							# assonance. i.e. '-ĭ':'-ī'
							alphabetized_match = sorted(['ϑ' + syllab[-1] for syllab in rich_ultim_syllabs])
							match = 'ççç'.join(alphabetized_match)
							rhymetype = 'asϑuς'
							rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
							bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
							results_dict[center_token].append(bundle)
					else:
						# consonance. i.e. '-m':'-m'
						alphabetized_match = sorted(['ϑ' + syllab[-1] for syllab in rich_ultim_syllabs])
						match = 'ççç'.join(alphabetized_match)
						rhymetype = 'consϑuς'
						rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
						bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
						results_dict[center_token].append(bundle)

				# ******************************
				# match on idcs -1, -2 (homoioteleuton)
				# match on idcs -1, -2 en -3 (homoioteleuton)
				if ultim_syllabs_corrs in [[-1, -2], [-1, -2, -3]]:
					alphabetized_match = sorted(rich_ultim_syllabs)
					match = 'ççç'.join(alphabetized_match)
					if at_sense_pause == True:
						rhymetype = 'pure-homς'
						rhyme_score = 1 * rhyme_probability # distance is not penalized, but rhyme probability should count...
					else:
						rhymetype = 'homς'
						rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
					bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
					results_dict[center_token].append(bundle)

				# =======================================
				# STEP 3 - detecting rhyme in penultima syllaba
				penultim_syllabs_corrs = syllab_matcher(penultim_syllabs)
				if len(penultim_syllabs_corrs) >= 2:
					alphabetized_match = sorted(rich_penultim_syllabs)
					match = 'ççç'.join(alphabetized_match)
					rhymetype = 'penultς'
					# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
					rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
					bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
					results_dict[center_token].append(bundle)

				# =======================================
				# STEP 4 - check for alliteration
				# match on idcs 0, 1
				try: # first letter match - idx on 0
					alliteration = 0
					if prim_syllabs[0][0] == prim_syllabs[1][0]:
						alliteration += 1
						try: # second letter match - idx on 1
							if prim_syllabs[0][1] == prim_syllabs[1][1]:
								alliteration += 1
						except IndexError:
							pass
					if alliteration > 0:
						alphabetized_match = sorted(rich_prim_syllabs)
						match = 'ççç'.join(alphabetized_match)
						rhymetype = 'allς'
						# rhyme_score = (sent_length / absolute_dist / sent_length) / sent_length # sense pause does not matter
						rhyme_score = (1 / absolute_dist) * rhyme_probability  # update 26-3-23
						bundle = (global_idx_center_token, target_token, match, rhymetype, rhyme_score)
						results_dict[center_token].append(bundle)
				except IndexError:
					pass
			
			for center_token, bundles in results_dict.items():
				for (global_idx_center_token, target_token, match, rhymetype, rhyme_score) in bundles:
					if center_token not in ['κμ']:
						new_fob.write(str(global_idx_center_token) + '\t' + center_token + '\t' + target_token + '\t' +  match  + '\t' + rhymetype + '\t' + str(rhyme_score) + '\n')


class RhymeBarplot():
	def __init__(self, folder_name):
		self.folder_name = folder_name

	def go(folder_name):

		lengths = {'Historia-translationis-sanctae-Lewinne': 10795,
			'Vita-S-Eadburge-Gotha': 1039,
			'Vita-Herlindis-et-Reinulae': 4357,
			'Vita-S-Eadburge-Hereford': 821,
			'combined-letters': 657,
			'Vita-Dunstani-Osberni': 20630,
			'Vita-Dunstani': 10418,
			'Vita-ss-Aethelredi-et-Aethelberti-martirum': 1512,
			'Miracula-Eadburge': 1782,
			'Vita-Mathildis-reginae-posterior': 10648,
			'Vita-et-miracula-Elphegi': 5864,
			'Vita-S-Ecgwini': 9209,
			'Vita-S-Ecgwini-Dom': 13715,
			'Miracula-S-Oswaldi': 4611,
			'Vita-S-Wilfridi': 15316,
			'Vita-S-Anselmi': 26927,
			'Vita-Odonis': 4063,
			'Vita-S-Dunstani': 15132,
			'Vita-S-Oswaldi-Byrthferthi': 17871,
			'Vita-S-Oswaldi-Eadmeri': 9388,
			'Vita-S-Bregowini': 2813,
			'Miracula-S-Dunstani': 6447,
			'Carmen-de-S-Vigore': 170,
			'De-translatione-sanctorum-qui-in-Thornensi-monasterio-requiescunt': 924,
			'Vita-S-Bertini': 5958,
			'Vita-sancti-Johannis-Beverlacensis': 4594,
			'De-sanctis-Thancredo-et-Torhtredo': 501,
			'Vita-Botulphi': 1689,
			'Vita-secunda-Remacli': 6214,
			'Encomium-Emmae-reginae': 8870,
			'Miracula-S-Eadmundi-regis-et-martyris': 15068,
			'Vita-tertia-S-Bovonis-Gandensis': 7033,
			'Vita-Rumoldi-Mechliniensis': 3403,
			'Relatio-de-inventione-et-elevatione-sancti-Bertini': 6998,
			'Vita-S-Aethelwoldi': 6853,
			'Translatio-et-miracula-S-Swithuni': 13199,
			'Historia-Normannorum': 51633,
			'Historia-Normannorum(no-verse)': 41377,
			'Vita-Amelbergae': 8548,
			'Miracula-Aetheldrethe': 4484,
			'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 8463,
			'Vita-Kenelmi': 3095,
			'Vita-Werburge': 2682,
			'In-natale-S-Edwoldi': 1195,
			'Translatio-Wulfildae': 973,
			'Vita-Letardi': 1886,
			'Vita-Milburge-ed-Love': 8570,
			'Vita-Kenelmi-brevior': 1036,
			'Vita-Wihtburge': 3605,
			'Miracula-S-Eadmundi-regis-et-martyris': 26567,
			'Visio-Alvivae': 817,
			'Inventio-Amelbergae': 1275,
			'Vita-Sexburge': 6803,
			'Vita-Milburge': 8556,
			'Vita-Aetheldrethe-recensio-C': 2283,
			'Lectiones-Sexburge': 832,
			'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 1271,
			'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 3045,
			'Vita-Aetheldrethe-recensio-D': 1513,
			'Translatio-Yvonis': 2924,
			'Miracula-Wihtburge': 4553,
			'Vita-Ædwardi-no-verse': 8609,
			'Lectiones-Eormenhilde': 1305,
			'Passio-Eadwardi-regis-et-martyris': 3614,
			'Lectiones-de-Hildelitha': 936,
			'Vita-Wlsini(not-amended)': 5012,
			'Vita-Wlsini(amended-Love)': 5013,
			'Translatio-Augustini-et-aliorum-sanctorum': 22599,
			'Vita-Vulfilda': 3540,
			'Vita-Edithae': 9909,
			'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 8061,
			'Historia-minor-Augustini': 7375,
			'Vita-Mildrethe': 10133,
			'Inventio-Yvonis[PL+Macray]': 7803,
			'Vita-Augustini': 16568,
			'Liber-confortatorius': 32815,
			'Vita-et-Miracula-Æthelburge': 4813,
			'Vita-Edithae-no-verse': 8448,
			'Translatio-Mildrethe': 15691,
			'Vita-de-Iusto-archiepiscopo': 1281,
			'Miracula-Augustini-maior': 12212,
			'Tomellus-Amelbergae': 1161,
			'Miracula-S-Winnoci': 6577,
			'Vita-S-Godelivae': 3727}

		colours_dict = {'B': '#ED6B54',
			'Bovo-Sithiensis': '#00ACFF',
			'Byrhtferth-of-Ramsey': '#FEAE00',
			'Dominic-of-Evesham': '#EAD4ED',
			'Drogo-Bergis-S-Winoci': '#D77DED',
			'Dudo-Viromandesis': '#A9EDFF',
			'Eadmerus': '#EC7FA0',
			'Encomiast': '#32EDBE',
			'Folcardus': '#9ECB8B',
			'Goscelinus': '#8BA5DF',
			'Goscelinus-maybe': 'r',
			'Heriger-of-Lobbes': '#FFE6B1',
			'Hermannus-archidiaconus': '#3F9EA3',
			'Lantfred-of-Winchester': '#DEEDAB',
			'Osbernus-Cantuariensis': '#EAD0FB',
			'Vita-Ædwardi': '#EDCFCD',
			'Wulfstan-of-Winchester': '#38ED2D'}

		school_dict = {'B': 'no',
			'Bovo-Sithiensis': 'yes',
			'Byrhtferth-of-Ramsey': 'no',
			'Dominic-of-Evesham': 'no',
			'Drogo-Bergis-S-Winoci': 'yes',
			'Dudo-Viromandesis': 'no',
			'Eadmerus': 'no',
			'Encomiast': 'yes',
			'Folcardus': 'yes',
			'Goscelinus': 'yes',
			'Goscelinus-maybe': 'yes',
			'Heriger-of-Lobbes': 'no',
			'Hermannus-archidiaconus': 'no',
			'Lantfred-of-Winchester': 'no',
			'Osbernus-Cantuariensis': 'no',
			'Vita-Ædwardi': 'yes',
			'Wulfstan-of-Winchester': 'no'}

		abbrev_titles = {'Vita-S-Godelivae': 'VGod',
			'Miracula-S-Eadmundi-regis-et-martyris': 'MEadm',
			'De-sanctis-Thancredo-et-Torhtredo': 'VThanTorh',
			'Historia-translationis-sanctae-Lewinne': 'HLew',
			'Vita-Dunstani': 'VDun',
			'Vita-Amelbergae': 'VAmel',
			'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 'VAmel',
			'Vita-S-Anselmi': 'VAns',
			'Miracula-S-Winnoci': 'MWin',
			'Vita-Vulfilda': 'VVulf',
			'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 'LibMild',
			'Vita-Ædwardi-no-verse': 'VÆdw',
			'Translatio-Mildrethe': 'TMild',
			'Vita-S-Wilfridi': 'VWilf',
			'Vita-S-Bregowini': 'VBreg',
			'Vita-et-Miracula-Æthelburge': 'VMÆth',
			'Vita-S-Oswaldi-Byrthferthi': 'VOsw',
			'Vita-Mildrethe': 'VMild',
			'Miracula-S-Oswaldi': 'MOsw',
			'Inventio-Yvonis[PL+Macray]': 'IYvo',
			'Miracula-S-Dunstani': 'MDun',
			'Vita-et-miracula-Elphegi': 'VMElph',
			'Miracula-Eadburge': 'MEadb',
			'Miracula-Augustini-maior': 'MAug',
			'Vita-Edithae-no-verse': 'VEd',
			'Encomium-Emmae-reginae': 'EncEmm',
			'Translatio-Augustini-et-aliorum-sanctorum': 'TAug',
			'Liber-confortatorius': 'LConf',
			'Vita-Wlsini(amended-Love)': 'VWls',
			'Historia-minor-Augustini': 'HAug-minor',
			'Vita-Augustini': 'VAug',
			'Vita-secunda-Remacli': 'VRemacl',
			'Vita-S-Dunstani': 'VDun',
			'Vita-S-Aethelwoldi': 'VAethelw',
			'Vita-S-Oswaldi-Eadmeri': 'VOsw',
			'De-translatione-sanctorum-qui-in-Thornensi-monasterio-requiescunt': 'TThorn',
			'Vita-S-Eadburge-Gotha': 'VEadb',
			'Vita-S-Ecgwini': 'VEcgw',
			'Vita-S-Ecgwini-Dom': 'VEcgw',
			'Relatio-de-inventione-et-elevatione-sancti-Bertini': 'InvBert',
			'Vita-de-Iusto-archiepiscopo': 'VIust',
			'Vita-Botulphi': 'VBot',
			'Vita-sancti-Johannis-Beverlacensis': 'VIoanBev',
			'Vita-Odonis': 'VOdo',
			'Translatio-et-miracula-S-Swithuni': 'TSwith',
			'Vita-S-Eadburge-Hereford': 'VEadb',
			'Historia-Normannorum(no-verse)': 'HNorman',
			'Vita-Herlindis-et-Reinulae': 'VHerRein',
			'Vita-ss-Aethelredi-et-Aethelberti-martirum': 'VAethAeth',
			'Vita-S-Bertini': 'VBert',
			'Vita-Dunstani-Osberni': 'VDun'}

		sent_len_normalizer = {'Vita-S-Godelivae': 17.013513513513512,
		'Carmen-de-S-Vigore': 170.0,
		'Miracula-S-Eadmundi-regis-et-martyris': 18.571629213483146,
		'Historia-translationis-sanctae-Lewinne': 18.797909407665504,
		'Miracula-Wihtburge': 18.970833333333335,
		'De-sanctis-Thancredo-et-Torhtredo': 20.04,
		'Miracula-S-Winnoci': 20.393846153846155,
		'Vita-Milburge-ed-Love': 20.45346062052506,
		'Vita-S-Anselmi': 22.121610517666394,
		'Vita-Rumoldi-Mechliniensis': 22.686666666666667,
		'Historia-Normannorum(no-verse)': 23.180392156862744,
		'Vita-secunda-Remacli': 23.698473282442748,
		'Vita-Mildrethe': 24.002369668246445,
		'Vita-Kenelmi': 24.1796875,
		'Translatio-Yvonis': 24.341666666666665,
		'Vita-S-Bregowini': 24.67543859649123,
		'Vita-S-Wilfridi': 24.736672051696285,
		'Vita-S-Eadburge-Hereford': 24.87878787878788,
		'Vita-Wlsini(not-amended)': 25.436548223350254,
		'Miracula-Augustini-maior': 25.65546218487395,
		'Translatio-Augustini-et-aliorum-sanctorum': 25.739179954441912,
		'Vita-Wlsini(amended-Love)': 25.968911917098445,
		'Vita-S-Eadburge-Gotha': 25.975,
		'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 26.036923076923078,
		'Vita-Kenelmi-brevior': 26.564102564102566,
		'Historia-minor-Augustini': 26.721014492753625,
		'Liber-confortatorius': 26.72231270358306,
		'Vita-Letardi': 26.942857142857143,
		'Translatio-Mildrethe': 27.129757785467127,
		'Vita-Amelbergae': 27.136507936507936,
		'De-translatione-sanctorum-qui-in-Thornensi-monasterio-requiescunt': 27.176470588235293,
		'Vita-Dunstani': 27.617135207496652,
		'Vita-et-miracula-Elphegi': 27.781990521327014,
		'Miracula-S-Dunstani': 27.90909090909091,
		'Vita-S-Ecgwini': 28.03030303030303,
		'Historia-Normannorum': 28.181768558951966,
		'Miracula-S-Oswaldi': 28.288343558282207,
		'Relatio-de-inventione-et-elevatione-sancti-Bertini': 28.450892857142858,
		'Vita-S-Oswaldi-Byrthferthi': 28.61698717948718,
		'Vita-Vulfilda': 29.016393442622952,
		'Tomellus-Amelbergae': 29.025,
		'Vita-S-Oswaldi-Eadmeri': 29.165644171779142,
		'Vita-et-Miracula-Æthelburge': 29.169696969696968,
		'Vita-Augustini': 29.220458553791886,
		'Vita-de-Iusto-archiepiscopo': 29.790697674418606,
		'Encomium-Emmae-reginae': 29.95945945945946,
		'Vita-sancti-Johannis-Beverlacensis': 30.423841059602648,
		'Vita-Botulphi': 30.70909090909091,
		'Vita-S-Ecgwini': 30.794642857142858,
		'Vita-ss-Aethelredi-et-Aethelberti-martirum': 30.836734693877553,
		'Miracula-S-Eadmundi-regis-et-martyris': 31.119834710743802,
		'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 31.4765625,
		'Vita-Sexburge': 31.49537037037037,
		'Inventio-Yvonis[PL+Macray]': 31.8,
		'Vita-S-Dunstani': 31.924050632911392,
		'Miracula-Eadburge': 32.38181818181818,
		'Inventio-Amelbergae': 32.69230769230769,
		'combined-letters': 32.85,
		'Vita-Odonis': 33.03252032520325,
		'Vita-tertia-S-Bovonis-Gandensis': 34.47549019607843,
		'Miracula-Aetheldrethe': 34.75968992248062,
		'Vita-Mathildis-reginae-posterior': 343.48387096774195,
		'Vita-Herlindis-et-Reinulae': 35.13709677419355,
		'Vita-Aetheldrethe-recensio-D': 35.18604651162791,
		'Visio-Alvivae': 35.52173913043478,
		'Vita-Ædwardi-no-verse': 36.02092050209205,
		'Vita-Aetheldrethe-recensio-C': 36.23809523809524,
		'Lectiones-Eormenhilde': 36.25,
		'Translatio-et-miracula-S-Swithuni': 36.96638655462185,
		'Translatio-Wulfildae': 37.42307692307692,
		'Lectiones-Sexburge': 37.81818181818182,
		'Vita-Edithae-no-verse': 37.83408071748879,
		'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 40.03947368421053,
		'Vita-Dunstani': 40.053639846743295,
		'Vita-Wihtburge': 40.05555555555556,
		'Vita-Werburge': 41.261538461538464,
		'Passio-Eadwardi-regis-et-martyris': 41.54022988505747,
		'Vita-Edithae': 43.03478260869565,
		'Vita-Ædwardi copy': 44.62845849802372,
		'Vita-S-Bertini': 45.48854961832061,
		'Vita-S-Aethelwoldi': 45.993288590604024,
		'Lectiones-de-Hildelitha': 46.8,
		'In-natale-S-Edwoldi': 47.8,
		'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 52.958333333333336,
		'Vita-Milburg': 8.695829094608342}

		data = {}
		for filename in glob.glob(folder_name+'/*'):
			metadata = filename.split('/')[-1].split('_')
			author = metadata[0]
			title = metadata[1].split('.')[0]
			
			all_cons = 0
			all_ass = 0
			all_allit = 0
			all_homs = 0
			all_penults = 0
			
			for line in open(filename):
				global_idx, center_token, target_token, match, rhymetype, rhymescore = line.split()
				rhymescore = float(rhymescore)
				if rhymetype == 'allς':
					all_allit += rhymescore
				elif rhymetype in ['homς', 'pure-homς']:
					all_homs += rhymescore
				elif rhymetype.split('ϑ')[0] == 'as':
					all_ass += rhymescore
				elif rhymetype.split('ϑ')[0] == 'cons':
					all_cons += rhymescore
				elif rhymetype == 'penultς':
					all_penults += rhymescore

			all_cons = all_cons/lengths[title]
			all_ass = all_ass/lengths[title]
			all_allit = all_allit/lengths[title]
			all_homs = all_homs/lengths[title]
			all_penults = all_penults/lengths[title]
			
			# print(author, title)
			# print('consonance: {}'.format(all_cons))
			# print('assonance: {}'.format(all_ass))
			# print('alliteration: {}'.format(all_allit))
			# print('homoioteleuton: {}'.format(all_homs))
			# print('penultimate rhyme: {}'.format(all_penults))
			# print()

			data[title] = ((author, lengths[title]), all_cons, all_ass, all_allit, all_homs, all_penults)

		# target_idx = 5

		# sort by the kind of rhyme type you are interested in!
		sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: np.mean(item[1][1:]))}
		# sorted_data = {k: v for k, v in sorted(data.items(), key=lambda item: np.sum(item[1][4]))} # if you want to sort on homoeoteleuta

		y_cons = [v[1] for k, v in sorted_data.items()]
		y_ass = [v[2] for k, v in sorted_data.items()]
		y_allit = [v[3] for k, v in sorted_data.items()]
		y_homs = [v[4] for k, v in sorted_data.items()]
		y_penults = [v[5] for k, v in sorted_data.items()]

		x = [idx+1 for idx, each in enumerate(y_cons)]

		x_colours = [colours_dict[v[0][0]] for k, v in sorted_data.items()]
		x_schools = [school_dict[v[0][0]] for k, v in sorted_data.items()]
		x_labels = [abbrev_titles[k] for k, v in sorted_data.items()]
		x_metadata = [v[0] for k, v in sorted_data.items()]

		# """
		# BARPLOT
		# """

		# font_dirs = ['/Users/...']
		# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

		# for font_file in font_files:
		#	 font_manager.fontManager.addfont(font_file)

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.family'] = 'ArnoPro-Display'

		italics = {'fontname':'ArnoPro-Italic'} # https://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python

		# fig = plt.figure(figsize=(5,8)) # uncomment to make vertical
		fig = plt.figure(figsize=(10,5)) # uncomment to make horizontal
		ax2 = fig.add_subplot(111)

		for xx, yy_homs, yy_allit, yy_ass, yy_cons, yy_penults, x_colour, (x_author, x_text_length), x_school in zip(x, y_homs, y_allit, y_ass, y_cons, y_penults, x_colours, x_metadata, x_schools):

			if x_author == 'Goscelinus':
				x_author = 'Goscelinus S. Bertini'
			elif x_author == 'Folcardus':
				x_author = 'Folcardus S. Bertini'
			elif x_author == 'B':
				x_author = 'B.'
			elif x_author == 'Eadmerus':
				x_author = 'Eadmerus-Cantuariensis'
			elif x_author == 'Wulfstan-of-Winchester':
				x_author = 'Wulfstanus Wintoniensis'
			elif x_author == 'Byrhtferth-of-Ramsey':
				x_author = 'Byrhtferð Ramesigus'
			elif x_author == 'Lantfred-of-Winchester':
				x_author = 'Lantfredus Wintoniensis'
			elif x_author == 'Vita-Ædwardi':
				x_author = 'The Anonymous'
			elif x_author == 'Drogo-Bergis-S-Winoci':
				x_author = 'Drogo-Bergis-S.-Winoci'
			elif x_author == 'Dominic-of-Evesham':
				x_author = 'Dominicus Eveshamiae'

			replace_pattern  = re.compile('-')

			x_author = re.sub(replace_pattern, ' ', x_author)
			# ax2.barh(xx, yy_homs, color=x_colour, alpha=1, zorder=5, edgecolor='k', linewidth=0.5)
			# ax2.barh(xx, yy_homs+yy_allit, color=x_colour, alpha=0.8, zorder=4, edgecolor='k', linewidth=0.5)
			# ax2.barh(xx, yy_homs+yy_allit+yy_ass, color=x_colour, alpha=0.6, zorder=3, edgecolor='k', linewidth=0.5)
			# ax2.barh(xx, yy_homs+yy_allit+yy_ass+yy_cons, color=x_colour, alpha=0.4, zorder=2, edgecolor='k', linewidth=0.5)
			# ax2.barh(xx, yy_homs+yy_allit+yy_ass+yy_cons+yy_penults, color=x_colour, alpha=0.2, zorder=1, edgecolor='k', linewidth=0.5)

			filter_list = ['Folcardus S. Bertini'] # manipulate visible bars with this list
			if x_author in filter_list:
				ax2.bar(xx, yy_homs, color=x_colour, alpha=1, zorder=5, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit, color=x_colour, alpha=0.8, zorder=4, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass, color=x_colour, alpha=0.6, zorder=3, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass+yy_cons, color=x_colour, alpha=0.4, zorder=2, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass+yy_cons+yy_penults, color=x_colour, alpha=0.2, zorder=1, edgecolor='k', linewidth=0.5)

				if x_school == 'yes':
					school_c = 'k'
				else:
					school_c = '#ffffff'

				# ax2.barh(xx, 0.1, color=school_c, alpha=0.8, left=1.5, height=0.8, zorder=0, edgecolor='k', linewidth=0.5) # uncomment when barchart is vertical
				ax2.bar(xx, 0.0005, color=school_c, alpha=0.8, bottom=0.0145, zorder=0, edgecolor='k', linewidth=0.5) # uncomment when barchart is horizontal
				# ax2.text(1.7, xx-0.35, x_author, alpha=0.9) # (x left begin, y position, text, alpha)
			else: # you need this to keep the figure in correct proportion
				ax2.bar(xx, yy_homs, color=x_colour, alpha=0, zorder=5, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit, color=x_colour, alpha=0, zorder=4, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass, color=x_colour, alpha=0, zorder=3, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass+yy_cons, color=x_colour, alpha=0, zorder=2, edgecolor='k', linewidth=0.5)
				ax2.bar(xx, yy_homs+yy_allit+yy_ass+yy_cons+yy_penults, color=x_colour, alpha=0, zorder=1, edgecolor='k', linewidth=0.5)

		# ax2.invert_yaxis()  # labels read top-to-bottom

		ax2.set_xticks(x) # uncomment to make barchart horizontal
		# ax2.set_yticks(x) # uncomment to make barchart vertical
		ax2.set_xticklabels(x_labels, **italics, rotation=70) # uncomment to make barchart horizontal
		# ax2.set_yticklabels(x_labels, **italics) # uncomment to make barchart vertical

		ax2.set_ylabel('Number of rhyme matches normalized by text length') # uncomment to make barchart horizontal
		# ax2.set_xlabel('Number of rhyme matches normalized by text length') # uncomment to make barchart vertical

		ax2.spines['top'].set_visible(False)
		ax2.spines['right'].set_visible(False)

		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/barplot-rhymes-7-4-onlyfolc.pdf", bbox_inches='tight', transparent=True, format='pdf')

class RhymeLineplot():
	def __init__(self, folder_location):
		self.folder_name = folder_location

	def go(folder_location):

		colours_dict = {'Vita-Mildrethe': '#90A4DB',
					    'Vita-Amelbergae-(Love-ed-Salisbury-witness)': '#C3DAFF'}

		lengths = {'Historia-translationis-sanctae-Lewinne': 10795,
			'Vita-S-Eadburge-Gotha': 1039,
			'Vita-Herlindis-et-Reinulae': 4357,
			'Vita-S-Eadburge-Hereford': 821,
			'combined-letters': 657,
			'Vita-Dunstani-Osberni': 20630,
			'Vita-Dunstani': 10418,
			'Vita-ss-Aethelredi-et-Aethelberti-martirum': 1512,
			'Miracula-Eadburge': 1782,
			'Vita-Mathildis-reginae-posterior': 10648,
			'Vita-et-miracula-Elphegi': 5864,
			'Vita-S-Ecgwini': 9209,
			'Vita-S-Ecgwini-Dom': 13715,
			'Miracula-S-Oswaldi': 4611,
			'Vita-S-Wilfridi': 15316,
			'Vita-S-Anselmi': 26927,
			'Vita-Odonis': 4063,
			'Vita-S-Dunstani': 15132,
			'Vita-S-Oswaldi-Byrthferthi': 17871,
			'Vita-S-Oswaldi-Eadmeri': 9388,
			'Vita-S-Bregowini': 2813,
			'Miracula-S-Dunstani': 6447,
			'Carmen-de-S-Vigore': 170,
			'De-translatione-sanctorum-qui-in-Thornensi-monasterio-requiescunt': 924,
			'Vita-S-Bertini': 5958,
			'Vita-sancti-Johannis-Beverlacensis': 4594,
			'De-sanctis-Thancredo-et-Torhtredo': 501,
			'Vita-Botulphi': 1689,
			'Vita-secunda-Remacli': 6214,
			'Encomium-Emmae-reginae': 8870,
			'Miracula-S-Eadmundi-regis-et-martyris': 15068,
			'Vita-tertia-S-Bovonis-Gandensis': 7033,
			'Vita-Rumoldi-Mechliniensis': 3403,
			'Relatio-de-inventione-et-elevatione-sancti-Bertini': 6998,
			'Vita-S-Aethelwoldi': 6853,
			'Translatio-et-miracula-S-Swithuni': 13199,
			'Historia-Normannorum': 51633,
			'Historia-Normannorum(no-verse)': 41377,
			'Vita-Amelbergae': 8548,
			'Miracula-Aetheldrethe': 4484,
			'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 8463,
			'Vita-Kenelmi': 3095,
			'Vita-Werburge': 2682,
			'In-natale-S-Edwoldi': 1195,
			'Translatio-Wulfildae': 973,
			'Vita-Letardi': 1886,
			'Vita-Milburge-ed-Love': 8570,
			'Vita-Kenelmi-brevior': 1036,
			'Vita-Wihtburge': 3605,
			'Miracula-S-Eadmundi-regis-et-martyris': 26567,
			'Visio-Alvivae': 817,
			'Inventio-Amelbergae': 1275,
			'Vita-Sexburge': 6803,
			'Vita-Milburge': 8556,
			'Vita-Aetheldrethe-recensio-C': 2283,
			'Lectiones-Sexburge': 832,
			'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 1271,
			'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 3045,
			'Vita-Aetheldrethe-recensio-D': 1513,
			'Translatio-Yvonis': 2924,
			'Miracula-Wihtburge': 4553,
			'Vita-Ædwardi-no-verse': 8609,
			'Lectiones-Eormenhilde': 1305,
			'Passio-Eadwardi-regis-et-martyris': 3614,
			'Lectiones-de-Hildelitha': 936,
			'Vita-Wlsini(not-amended)': 5012,
			'Vita-Wlsini(amended-Love)': 5013,
			'Translatio-Augustini-et-aliorum-sanctorum': 22599,
			'Vita-Vulfilda': 3540,
			'Vita-Edithae': 9909,
			'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 8061,
			'Historia-minor-Augustini': 7375,
			'Vita-Mildrethe': 10133,
			'Inventio-Yvonis[PL+Macray]': 7803,
			'Vita-Augustini': 16568,
			'Liber-confortatorius': 32815,
			'Vita-et-Miracula-Æthelburge': 4813,
			'Vita-Edithae-no-verse': 8448,
			'Translatio-Mildrethe': 15691,
			'Vita-de-Iusto-archiepiscopo': 1281,
			'Miracula-Augustini-maior': 12212,
			'Tomellus-Amelbergae': 1161,
			'Miracula-S-Winnoci': 6577,
			'Vita-S-Godelivae': 3727}

		sample_len = 300
		X = []
		Y = []
		titles = []

		for filename in glob.glob(folder_location + "/*"):

			metadata = filename.split('/')[-1].split('_')
			author = metadata[0]
			title = metadata[1].split('.')[0]
			ranges = [range(i, i+sample_len) for i in range(0, lengths[title], sample_len)]
			titles.append(title)

			results_dict = {}
			for each_range in ranges:
				right_stop = each_range[-1]+1
				results_dict[right_stop] = 0

			for line in open(filename):
				global_idx, center_token, target_token, match, rhymetype, rhymescore = line.split()
				global_idx = int(global_idx)
				for each_range in ranges:
					right_stop = each_range[-1]+1
					if global_idx in each_range:
						results_dict[right_stop] += float(rhymescore)

			x = []
			y = []
			for (xx, yy) in results_dict.items():
				x.append(xx)
				y.append(yy)
			X.append(x)
			Y.append(y)

		# Layout elements
		
		# font_dirs = ['/Users/...']
		# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
		# for font_file in font_files:
		#	 font_manager.fontManager.addfont(font_file)

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.family'] = 'ArnoPro-Display'

		italics = {'fontname':'ArnoPro-Italic'} # https://stackoverflow.com/questions/21321670/how-to-change-fonts-in-matplotlib-python

		fig = plt.figure(figsize=(10,3))
		# fig = plt.figure(figsize=(3.5,2))
		ax = fig.add_subplot(111)

		ax.set_xlabel('Progression of text')
		ax.set_ylabel('Rhyme score')
		
		# # Despine
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['left'].set_visible(True)
		ax.spines['bottom'].set_visible(True)

		# Set ticks of indices
		# xticks = [1, 1255, 2463, 3185, 4382, 5085, 5733, 6721]
		# xlabels = ['i.1', 'i.3', 'i.4', 'i.5', 'i.6', 'Extra muros [...]', 'i.7', 'ii.1–3 + ii.11']

		for x, y, title in zip(X, Y, titles):
			ax.plot(x, y, linewidth=7, alpha=0.7, markersize=1.5, linestyle='-', color=colours_dict[title])

		ax.tick_params(axis='both', which='major', labelsize=8)
		# plt.xticks(xticks, xlabels, rotation='horizontal')
		# plt.yticks(yticks, ylabels, zorder=0)
		# ax.set_xlabel(xlabels, rotation=0, fontsize=20, labelpad=20)
		# ax.set_xlim(xmin=0, xmax=7273)
		plt.tight_layout()
		plt.show()
		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/rhymelineplot-try.pdf", transparent=True, format='pdf')
			
class RhymeVectorizer():
	def __init__(self, folder_location):
		self.folder_name = folder_location

	def go(folder_location):

		sample_len = 1000
		n_feats = 20

		data = {} # data['Goscelinus'] = {'Vita-Wlsini(amended-Love)': [], 'Translatio-Augustini-et-aliorum-sanctorum': [], ...}
		for filename in glob.glob(folder_location+'/*'):
			metadata = filename.split('/')[-1].split('_')
			author = metadata[0]
			data[author] = {}

		for filename in glob.glob(folder_location+'/*'):
			metadata = filename.split('/')[-1].split('_')
			author = metadata[0]
			title = metadata[1].split('.')[0]
			data[author].update({title: []})

		for filename in glob.glob(folder_location+'/*'):
			metadata = filename.split('/')[-1].split('_')
			author = metadata[0]
			title = metadata[1].split('.')[0]
			for line in open(filename):
				global_idx, center_token, target_token, match, rhymetype, rhymescore = line.split()
				global_idx = int(global_idx)
				rhymescore = float(rhymescore)

				_included = ['homς', 'consϑuς', 'pure-homς', 'nearϑasϑpς', 'allς', 'asϑuς', 'asϑpς', 'penultς'] # keep all types here
				_filter = ['homς', 'pure-homς'] # use this list to filter

				if rhymetype in _included:

					token = match.split('ççç') # rhyme match token
					token.insert(0, rhymetype)
					no_diacritics_token = remove_diacritics(token) # gets rid of diacritics

					a = list(no_diacritics_token[1])
					b = list(no_diacritics_token[2])
					
					# find matches at end of string
					reversed_s = []
					reversed_s_coded = []
					for char_a, char_b in zip(reversed(a), reversed(b)):
						if char_a == char_b:
							if char_a == 'ϑ':
								reversed_s_coded.append(0)
							else:
								reversed_s.append(char_a)
								reversed_s_coded.append(1)
						else:
							reversed_s.append('λ')
							reversed_s_coded.append(0)

					# find matches at start of string
					s = []
					s_coded = []
					for char_a, char_b in zip(a, b):
						if char_a == char_b:
							if char_a == 'ϑ':
								s_coded.append(0)
							else:
								s.append(char_a)
								s_coded.append(1)
						else:
							s.append('λ')
							s_coded.append(0)

					if np.sum(s_coded) > np.sum(reversed_s_coded):
						true_list = s
					else:
						true_list = list(reversed(reversed_s))
					true_list = [char for char in true_list if char != 'λ']

					universal_match = [''.join(true_list) for i in match.split('ççç')] # from ['scrīϑ', 'stŭϑ'] returns ['s', 's']
					# tie it together and return
					universal_match.insert(0, rhymetype)
					universal_token = universal_match

					# token = 'ççç'.join(universal_token) # if you want universal rhymes: ['s', 's'] instead of ['scrīϑ', 'stŭϑ']
					token = 'ççç'.join(token) # uncomment to go back to particulars
					tup = (global_idx, token, rhymescore)
					data[author][title].append(tup)

		vocabulary = {}
		for author, d in data.items(): # author = 'Bovo-Sithiensis'
			for title, tokens in d.items(): # tokens = [(1713, 'homςçççϑmçççϑm'), (1713, 'homςçççϑemçççϑem'), ...]
				for global_idx, feat, rhymescore in tokens:
					vocabulary[feat] = 0
		
		for author, d in data.items(): # author = 'Bovo-Sithiensis'
			for title, tokens in d.items(): # tokens = [(1713, 'homςçççϑmçççϑm'), (1713, 'homςçççϑemçççϑem'), ...]
				for global_idx, feat, rhymescore in tokens:
					# vocabulary[feat] += 1 # uncomment in case of working with raw tallies
					vocabulary[feat] += rhymescore # uncomment in case of working with normalized counts

		# Sort the dictionary by its values
		sorted_vocab = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

		all_feats = list(sorted_vocab)
		top_feats = all_feats[:n_feats]

		all_authors = []
		all_titles = []
		all_vecs = []
		for author, d in data.items(): # author = 'Bovo-Sithiensis'
			for title, tokens in d.items(): # tokens = [(1713, 'homςçççϑmçççϑm', 0.005), (1713, 'homςçççϑemçççϑem', 0.005), ...]
				all_idcs = []
				for global_idx, feat, rhymescore in tokens:
					all_idcs.append(global_idx)

				# build sample ranges from extant 'global' indices 
				sample_ranges = [range(i, i+sample_len) for i in range(0, all_idcs[-1], sample_len)]
				sample_titles = [title + '_' + str(idx) for idx, i in enumerate(sample_ranges)]

				for sample_title, sample_range in zip(sample_titles, sample_ranges):
					
					all_titles.append(sample_title)
					all_authors.append(author)

					vec = {feat: 0 for feat in top_feats}
					for (idx, token, rhymescore) in tokens:
						if idx in sample_range:
							if token in top_feats:
								vec[token] += rhymescore # uncomment in case of working with normalized rhyme scores
								# vec[token] += 1 # uncomment in case of working with raw tallies
					vec = [freq for feat, freq in vec.items()]
					all_vecs.append(vec)
		
		data = np.array(all_vecs)
		
		scaler = StandardScaler()
		X = scaler.fit_transform(data)
		authors = all_authors
		titles = all_titles
		
		return X, authors, titles, top_feats

class NetworkAnalysis():

	def __init__(self, X, authors, titles):
		self.X = X
		self.authors = authors
		self.titles = titles

	def go(X, authors, titles):

		n_nbrs = 21 # this value minus one for 'actual' n_nbrs
		# 3 neighbors for each sample is argued to make up enough consensus
		# Try to make a consensus of distance measures
		# Use cosine, euclidean and manhattan distance, and make consensus tree (inspired by Eder)

		metric_dictionary = {'manhattan': 'manhattan', 'cosine': 'cosine', 'euclidean': 'euclidean'}
		# metric_dictionary = {'euclidean': 'euclidean'}

		# initiating files for nodes and edges
		fob_nodes = open("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/readout/gephi/nodes.csv", "w")
		fob_edges = open("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/readout/gephi/edges.csv", "w")

		fob_nodes.write("Id" + "\t" + "Work" + "\t" + "Author" + "\t" + "School" + "\n")
		fob_edges.write("Source" + "\t" + "Target" + "\t" + "Type" + "\t" + "Weight" + "\n")

		# Build up consensus distances of different feature ranges and different metrics
		exhsearch_data = []

		for metric in tqdm(metric_dictionary, postfix='metric'):
			model = NearestNeighbors(n_neighbors=n_nbrs,
									algorithm='auto',
									metric=metric_dictionary[metric],
									).fit(X)
			# index = closest vector in data set
			distances, indices = model.kneighbors(X)
			
			# first nearest neighbour is identical to current sample, so we clip it off
			all_distances = []
			for distance_vector in distances:
				all_distances.append(distance_vector[1:])

			scaler = MinMaxScaler()
			all_distances = scaler.fit_transform(all_distances)
			
			# Distances appended to dataframe
			for distance_vec, index_vec in zip(all_distances, indices):
				# Instantiate number of requested neighbours
				current_sample_title = titles[index_vec[0]]
				
				# collect indices and titles of x closest neighbours
				target_idcs = index_vec[1:]
				closest_nbrs = [titles[i] for i in target_idcs]

				# Instantiate tuple consisting of parameters, current sample, and closest neighbours
				data_tup = ('{}'.format(metric_dictionary[metric]), current_sample_title)
				for title, distance in zip(closest_nbrs, distance_vec):
					data_tup = data_tup + (title,)
					# data_tup = data_tup + (distance[0],)
					data_tup = data_tup + (distance,)

				exhsearch_data.append(data_tup)

		# Construct columns for dataframe
		# number of neighbours always minus one because of zero index
		columns = ['exp', 'node']
		for i in range(0, n_nbrs-1):
			idx_string = str(i+1)
			neighbor = 'neighbor {}'.format(idx_string)
			dst = 'dst {}'.format(idx_string)
			columns.append(neighbor)
			columns.append(dst)

		df = pd.DataFrame(exhsearch_data, columns=columns).sort_values(by='node', ascending=0)
		final_data = []
		weights= []
		node_orientation  = {title: idx+1 for idx, title in enumerate(titles)}
		for idx, (author, title) in enumerate(zip(authors, titles)):
			neighbors = []
			dsts = []
			# Pool all neighbors and distances together (ignore ranking of nb1, nb2, etc.)
			for num in range(1, n_nbrs):
				neighbors.append([neighb for neighb in df[df['node']==title]['neighbor {}'.format(str(num))]])
				dsts.append([neighb for neighb in df[df['node']==title]['dst {}'.format(str(num))]])
			neighbors = sum(neighbors, [])
			dsts = sum(dsts, [])

			model = CountVectorizer(lowercase=False, token_pattern=r"[^=]*")
			count_dict = model.fit_transform(neighbors)
			names = [i for i in model.get_feature_names() if i != '']
			
			# Collect all the candidates per sample that were chosen by the algorithm as nearest neighbor at least once
			candidate_dict = {neighbor: [] for neighbor in names}
			for nbr, dst in zip(neighbors, dsts):
				candidate_dict[nbr].append(dst)
			candidate_dict = {nbr: np.mean(candidate_dict[nbr])*len(candidate_dict[nbr]) for nbr in candidate_dict}
			selection = sorted(candidate_dict.items(), key=lambda x: x[1], reverse=True)

			if author in ['Drogo-Bergis-S-Winoci', 'Goscelinus', 'Encomiast', 'Folcardus', 'Bovo-Sithiensis', 'Vita-Ædwardi']:
				school_boolean = 1
			else:
				school_boolean = 0
			fob_nodes.write(str(idx + 1) + "\t" + str(title.split('_')[0]) + "\t" + str(author) + '\t' + str(school_boolean) + "\n")
			data_tup = (title,)
			for candtitle, weight in selection[:n_nbrs-1]:
				data_tup = data_tup + (candtitle, weight,)
				weights.append(weight)
				fob_edges.write(str(idx+1) + "\t" + str(node_orientation[candtitle]) + "\t" + "Undirected" + "\t" + str(weight) + "\n")
			final_data.append(data_tup)

		# Prepare column names for dataframe
		longest = np.int((len(final_data[np.argmax([len(i) for i in final_data])]) - 1) / 2)
		columns = sum([['neighbor {}'.format(str(i)), 'dst {}'.format(str(i))] for i in range(1, longest+1)], [])
		columns.insert(0, 'node')
		final_df = pd.DataFrame(final_data, columns=columns).sort_values(by='node', ascending=0)
		print(final_df)

		return np.var(np.array(weights))

class RhymePCA():

	def __init__(self, X, authors, titles, features):
		self.X = X
		self.authors = authors
		self.titles = titles
		self.features = features

	def go(X, authors, titles, features):

		# block of code to make rhyme mathces more readable on the PCA plot
		# gets rid of Greek letters etc.
		nomenclature = {'homς': 'hom',
				'consϑuς': 'cons',
				'pure-homς': 'hom',
				'nearϑasϑpς': 'as',
				'allς': 'all', 
				'asϑuς': 'as', 
				'asϑpς': 'as',
				'penultς': 'pen'}
		new_fs = []
		for feat in features: 
			feat = feat.split('ççç')
			rhymetype = nomenclature[feat[0]]
			match_one = feat[1]
			match_two = feat[2]
			visualizable_feat = '{} : {} ({})'.format(match_one, match_two, rhymetype)
			new_fs.append(visualizable_feat)
		features = new_fs

		print(features[:20])

		pca = PCA(n_components=3)

		include = ['Goscelinus', 'Drogo-Bergis-S-Winoci']

		_filter = []
		for author in authors:
			if author in include:
				_filter.append(True)
			else:
				_filter.append(False)

		filtered_X = list(compress(X, _filter))
		filtered_authors = list(compress(authors, _filter))
		filtered_titles = list(compress(titles, _filter))
		
		X_bar = pca.fit_transform(filtered_X)
		var_exp = pca.explained_variance_ratio_

		var_pc1 = np.round(var_exp[0]*100, decimals=2)
		var_pc2 = np.round(var_exp[1]*100, decimals=2)
		var_pc3 = np.round(var_exp[2]*100, decimals=2)
		explained_variance = np.round(sum(pca.explained_variance_ratio_)*100, decimals=2)

		loadings = pca.components_.transpose()
		vocab_weights_p1 = sorted(zip(features, loadings[:,0]), \
								  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p2 = sorted(zip(features, loadings[:,1]), \
						   		  key=lambda tup: tup[1], reverse=True)
		vocab_weights_p3 = sorted(zip(features, loadings[:,2]), \
						   		  key=lambda tup: tup[1], reverse=True)

		print("Explained variance: ", explained_variance)
		print("Number of words: ", len(features))
		
		# Line that calls font
		# Custom fonts should be added to the matplotlibrc parameter .json files (fontlist-v310.json): cd '/Users/...'.
		# Make sure the font is in the font "library" (not just font book!)
		# Also: you would want to change the 'name : xxx' entry in the fontlist-v310.json file.
		# Documentation on changing font (matplotlibrc params): http://www.claridgechang.net/blog/how-to-use-custom-fonts-in-matplotlib

		rcParams['font.family'] = 'sans-serif'
		rcParams['font.family'] = ['Arno Pro']

		customized_colors = {'Eadmerus': '#DF859F',
							 'Encomiast': '#3F9EA3',
							 'Folcardus': '#9ECB8B',
							 'Goscelinus': '#90A4DB',
							 'Drogo-Bergis-S-Winoci': '#DFC89F',
							 'Hermannus-archidiaconus': '#3F9EA3',
							 'B': '#4A94BA',
							 'Bovo-Sithiensis': '#4A94BA',
							 'Byrhtferth-of-Ramsey': '#BC6F36',
							 'Dudo-Viromandesis': '#BC6F36',
							 'Lantfred-of-Winchester': '#C79491',
							 'Wulfstan-of-Winchester': '#A2D891',
							 'Vita-et-Inventio-Amelbergae': '#F5B97F',
							 'Vita-Ædwardi': 'r',
							 'VHerRein': '#C4D200',
							 'Heriger-of-Lobbes': '#3CA56A',
							 'The-Destruction-of-Troy': 'r',
							 'In-Cath-Catharda': 'b',
							 'VAmelb': 'r',
							 'Theodericus-Trudonensis': 'r'}
		replace_dict = {'Historia-minor-Augustini': 'HAug-min',
						'Inventio-Yvonis[PL+Macray]': 'InvYv',
						'Libellus-contra-inanes-sanctae-virginis-Mildrethae-usurpatores': 'Libellus',
						'Liber-confortatorius': 'LC',
						'Miracula-Augustini-maior': 'MAug-mai',
						'Translatio-Augustini-et-aliorum-sanctorum': 'TAug',
						'Translatio-Mildrethe': 'TMild',
						'Vita-Augustini': 'VAug',
						'Vita-Edithae-no-verse': 'VEdith',
						'Vita-Mildrethe': 'VMild',
						'Vita-Vulfilda': 'VVulf',
						'Vita-Wlsini(amended-Love)': 'VWls',
						'Vita-de-Iusto-archiepiscopo': 'VIust',
						'Vita-et-Miracula-Æthelburge': 'VÆthelb',
						'In-natale-S-Edwoldi': 'NatEdw',
						'Lectiones-Eormenhilde': 'LEorm',
						'Lectiones-Sexburge': 'LSex',
						'Lectiones-de-Hildelitha': 'LHild',
						'Miracula-Aetheldrethe': 'MÆtheld',
						'Miracula-S-Eadmundi-regis-et-martyris': 'MEadm',
						'Miracula-Wihtburge': 'MWiht',
						'Passio-Eadwardi-regis-et-martyris': 'PEadw',
						'Translatio-Ethelburgae-Hildelithae-Vlfhildae-maior': 'TÆthelb-mai',
						'Translatio-Ethelburgae-Hildelithae-Vlfhildae-minor': 'TÆthelb-min',
						'Translatio-Wulfildae': 'TWulf',
						'Translatio-Yvonis': 'TYv',
						'Visio-Alvivae': 'VisAlv',
						'Vita-Aetheldrethe-recensio-C': 'VÆtheld-c',
						'Vita-Aetheldrethe-recensio-D': 'VÆtheld-d',
						'Vita-Amelbergae-(Love-ed-Salisbury-witness)': 'VAmel',
						'Vita-Kenelmi-brevior': 'VKen-brev',
						'Vita-Kenelmi': 'VKen',
						'Vita-Letardi': 'VLet',
						'Vita-Milburge-ed-Love': 'VMilb',
						'Vita-Milburge': 'VMilb',
						'Vita-Sexburge': 'VSex',
						'Vita-Werburge': 'VWer',
						'Vita-Wihtburge': 'VWiht',
						'Vita-et-Inventio-Amelbergae': 'VAmel',
						'Vita-Ædwardi copy': 'VÆdwR',
						'Vita-Ædwardi-no-verse': 'VÆdwR',
						'Vita-S-Bertini': 'VBert',
						'Vita-sancti-Johannis-Beverlacensis': 'VIoan',
						'Vita-Botulphi': 'VBot',
						'merger-Thorney': 'Thorn'}

		fig = plt.figure(figsize=(5,3.2))
		ax = fig.add_subplot(111, projection='3d')
		
		x1, x2, x3 = X_bar[:,0], X_bar[:,1], X_bar[:,2]

		# Plot loadings in 3D
		l1, l2, l3 = loadings[:,0], loadings[:,1], loadings[:,2]

		scaler_one = MinMaxScaler(feature_range=(min(x1), max(x1)))
		scaler_two = MinMaxScaler(feature_range=(min(x2), max(x2)))
		scaler_three = MinMaxScaler(feature_range=(min(x3), max(x3)))

		realigned_l1 = scaler_one.fit_transform(l1.reshape(-1, 1)).flatten()
		realigned_l2 = scaler_two.fit_transform(l2.reshape(-1, 1)).flatten()
		realigned_l3 = scaler_three.fit_transform(l3.reshape(-1, 1)).flatten()
			
		# Makes the opacity of plotted features work
		abs_l1 = np.abs(l1)
		abs_l2 = np.abs(l2)
		abs_l3 = np.abs(l3)
		normalized_l1 = (abs_l1-min(abs_l1))/(max(abs_l1)-min(abs_l1))
		normalized_l2 = (abs_l2-min(abs_l2))/(max(abs_l2)-min(abs_l2))
		normalized_l3 = (abs_l3-min(abs_l3))/(max(abs_l3)-min(abs_l3))

		normalized_vocab_weights_p1 = sorted(zip(features, normalized_l1), \
								  key=lambda tup: tup[1], reverse=True)
		normalized_vocab_weights_p2 = sorted(zip(features, normalized_l2), \
						   		  key=lambda tup: tup[1], reverse=True)
		normalized_vocab_weights_p3 = sorted(zip(features, normalized_l3), \
						   		  key=lambda tup: tup[1], reverse=True)

		# Each feature's rank of importance on each of the PC's is calculated
		# Normalized by importance of PC
		d = {}
		for (feat, weight) in normalized_vocab_weights_p1:
			d[feat] = []
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p1):
			d[feat].append(idx * var_pc1)
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p2):
			d[feat].append(idx * var_pc2)
		for idx, (feat, weight) in enumerate(normalized_vocab_weights_p3):
			d[feat].append(idx * var_pc3)

		n_top_discriminants = 20 # adjust to visualize fewer or more discriminants
		best_discriminants = sorted([[feat, np.average(ranks)] for [feat, ranks] in d.items()], key = lambda x: x[1])
		top_discriminants = [i[0] for i in best_discriminants][:n_top_discriminants]

		# Scatterplot of datapoints
		for index, (p1, p2, p3, a, title) in enumerate(zip(x1, x2, x3, filtered_authors, filtered_titles)):

			full_title = title.split('_')[0]
			sample_number = title.split('_')[-1]
			# abbrev = replace_dict[full_title] + '-' + sample_number

			markersymbol = 'o'
			markersize = 20

			ax.scatter(p1, p2, p3, marker='o', color=customized_colors[a], s=markersize, zorder=1, alpha=1)

		# Plot features
		for x, y, z, opac_l1, opac_l2, opac_l3, feat, in zip(realigned_l1, realigned_l2, realigned_l3, normalized_l1, normalized_l2, normalized_l3, features):
			total_opac = (opac_l1 + opac_l2 + opac_l3)/3
			if feat in top_discriminants:
				ax.text(x, y, z, feat, color='k', ha='center', va="center", fontdict={'size': 17*total_opac}, zorder=10000, alpha=total_opac)

		# Important to adjust margins first when function words fall outside plot
		# This is due to the axes aligning (def align).
		# ax2.margins(x=0.15, y=0.15)

		ax.set_xlabel('PC 1: {}%'.format(var_pc1))
		ax.set_ylabel('PC 2: {}%'.format(var_pc2))
		ax.set_zlabel('PC 3: {}%'.format(var_pc3))

		plt.tight_layout()
		plt.show()

		fig.savefig("/Users/jedgusse/Documents/UGent - Lokaal/Conferences/Oxford - Goscelin/results/figures/Drogo-Bergis-S-Winoci.png", dpi=300, transparent=True, format='png', bbox_inches='tight')


# folder_location = '/Users/...'
# folder_location = '/Users/...'
# folder_location = '/Users/...'
# folder_location = '/Users/...'
# folder_location = '/Users/...'

# CollatinusPreprocess.go()
# result_file = open('/Users/...')
# FileSplitter.split()

# model_location = '/Users/...'
# test_folder_location = '/Users/...'
# train_folder_location = '/Users/...'

# alignment_table_location = '/Users/...'
# rhymes_folder_location = '/Users/...'

# PrinCompAnal.plot(folder_location, sample_len)
# FeatureSelection.go_chi2(folder_location)
# SVM_benchmarking.go(folder_location)
# SVM_db_visualization.go()
# RollingAnalysis.roll(test_folder_location, model_location)
# CollateTexts.go(folder_location)
# CollationAnalysis.go(alignment_table_location)
# LinearRegressionClass.go(folder_location)

# folder_location = '/Users/...'
# for author_folder in glob.glob(folder_location + "/*"):
# 	for filename in glob.glob(author_folder + "/*"):
# 		RhymeDetector.go(filename) # don't forget to feed tagged files

folder_location = '/Users/...'
# RhymeBarplot.go(folder_location)
X, authors, titles, features = RhymeVectorizer.go(folder_location)

# X = np.array([[3, 3, 1, 4], [4, 4, 2, 2], [7,7,2,3], [7,7,2,1], [3,3,2, 1]])
# authors = ['Folcardus', 'Goscelinus', 'Folcardus', 'Folcardus', 'Goscelinus']
# titles = ['Vita_1', 'Vita_2', 'Vita_3', 'Vita_4', 'Vita_5']

NetworkAnalysis.go(X, authors, titles)
# RhymePCA.go(X, authors, titles, features) # filter the authors you want to show in the class itself

# folder_location = '/Users/...'
# RhymeLineplot.go(folder_location)


