# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:59:21 2017

@author: tuur
"""
from __future__ import print_function
from dateutil import parser as dparser
from lib.evaluation import get_selective_rel_metrics, get_acc_from_confusion_matrix,save_confusion_matrix_from_metrics, viz_docs_rel_difference, save_entity_error_analysis
import random, re, os, shutil, time, datetime, pickle
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
import torch
from lib.data import reverse_dict_list
from lib.timeml import write_timebank_folder, get_dur_from_value
from lib.transformer.SubLayers import MultiHeadAttention
import itertools
from copy import copy
from collections import Counter, OrderedDict
import subprocess
from gensim.models.keyedvectors import KeyedVectors
from lib.yellowfin import YFOptimizer

random.seed(0)

torch.backends.cudnn.enabled=True


class TimelineModel(object):

	def setup_vocabularies(self, data, unk_threshold, special_conflation=False, entity_sequence=False):
		# Sets up indices for characters, POS, and words
		if entity_sequence:
			self.word_frequencies = Counter([token if not special_conflation else self.conflate_digits(token) for text in data for token in text.entity_tokens]) 		
		else:
			self.word_frequencies = Counter([token if not special_conflation else self.conflate_digits(token) for text in data for token in text.tokens]) 
		if unk_threshold:
			self.word_frequencies = Counter({token for token in self.word_frequencies if self.word_frequencies[token] > unk_threshold})
		
		all_features = set([f for doc in data for tok_index in range(len(doc.tokens)) for f in self.get_features(tok_index, doc)])
			
		cindex = {c:autograd.Variable(torch.from_numpy(np.array([i]))) for i,c in enumerate(set([c for w in self.word_frequencies for c in w]).union([self.unk_token]).union([str(n) for n in range(10)]))}
		pindex = {p:autograd.Variable(torch.from_numpy(np.array([i]))) for i,p in enumerate(set([p for text in data for p in text.pos] + [self.unk_token]))}	
		windex = {w:autograd.Variable(torch.from_numpy(np.array([i]))) for i,w in enumerate(list(self.word_frequencies.keys()) + [self.unk_token])}
		findex = {f:i for i,f in enumerate(list(all_features))}
		return windex, cindex, pindex, findex


	def get_params_from_nn_dict(self, nn_dict):
		params = []
		for name, component in nn_dict.items():
			params += self.get_component_params(name, nn_dict)
		return params

	def get_component_params(self, name, component_dict):
		if name in component_dict:
			component = component_dict[name]
			if hasattr(component, 'parameters'):
				return list(component.parameters())
			else:
				return [component] 

	def fix_component_by_name(self, name):
		component_names = [name] if name in self.nn else self.nn_by_subtask[name]
		for component_name in component_names:
			for par in self.get_component_params(component_name, self.nn):
				par.requires_grad=False
			self.tied_components.add(component_name)
	
			
	def free_component_by_name(self, name):
		component_names = [name] if name in self.nn else self.nn_by_subtask[name]
		for component_name in component_names:
			for par in self.get_component_params(component_name, self.nn):
				par.requires_grad=True
			if component_name in self.tied_components:
				self.tied_components.remove(component_name)


	def print_gradient_by_name(self, name=None):
		if name is None:
			components = self.nn.keys()
		else:
			components = [name] if name in self.nn else self.nn_by_subtask[name]

		for component in components:

			params = self.get_component_params(component, self.nn)
			summed = 0
			n_params = 0
			for p in params:
				if not p.grad is None:
					n_params += np.prod(list(p.size()))
					summ = sum(torch.abs(p.grad))
					if summ.size()[0] > 1:
						summ = sum(summ)
					summed += summ
			summed_grad = summed.data[0] if not type(summed)==int else summed
			print(component, round(summed_grad,2), '/',round(n_params,2),'=',round(float(summed_grad)/(n_params+1),2))
			

	def get_trainable_params(self):

		pars = set()
		for task in self.active_subtasks:
			component_names = self.nn_by_subtask[task]
			for comp in component_names:
				if comp in self.tied_components:
					continue

				for par in self.get_component_params(comp, self.nn):
					if par is not None and par.requires_grad:
						pars.add(par)
		return pars

	def reset_optimizer(self):
		trainable_params = self.get_trainable_params()
		if self.optimizer_type == 'adam':
			self.optimizer = optim.Adam(trainable_params, lr=self.lr)
		if self.optimizer_type == 'adaml2':
			self.optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=0.0001)
		if self.optimizer_type == 'amsgrad':
			self.optimizer = optim.Adam(trainable_params, lr=self.lr, amsgrad=True)
		if self.optimizer_type == 'amsgrad0.01':
			self.optimizer = optim.Adam(trainable_params, lr=self.lr, amsgrad=True, eps=0.01)
		if self.optimizer_type == 'amsgrad0.001':
			self.optimizer = optim.Adam(trainable_params, lr=self.lr, amsgrad=True, eps=0.001)			
		elif self.optimizer_type== 'adadelta':
			self.optimizer = optim.Adadelta(trainable_params, lr=self.lr)
		elif self.optimizer_type == 'rmsprop':
			self.optimizer = optim.RMSprop(trainable_params, lr=self.lr)
		elif self.optimizer_type == 'sgd':
			self.optimizer = optim.SGD(trainable_params, lr=self.lr, momentum=0.9, weight_decay=0.001)	
		elif self.optimizer_type == 'nesterov':
			self.optimizer = optim.SGD(trainable_params, lr=self.lr, momentum=0.9, weight_decay=0.001, nesterov=True)	
		elif self.optimizer_type == 'asgd':
			self.optimizer = optim.ASGD(trainable_params, lr=self.lr)
		elif self.optimizer_type == 'yf':
			self.optimizer = YFOptimizer(trainable_params)
			

	def move_to_gpu(self):
		for cname, component in self.nn.items():
			if hasattr(component, 'data'):
				component.data = component.data.cuda()
			else:
				component = component.cuda()

		for cname, constant in self.constants.items():
			constant.data = constant.data.cuda()
		for indices in [self.windex, self.pindex, self.cindex]:
			for w,i in indices.items():
				indices[w] = indices[w].cuda()
	

	def get_features(self, w_index, doc):
		w_span = doc.spans[w_index]
		annotations = doc.reverse_span_annotations[w_span] if w_span in doc.reverse_span_annotations else []
		features = []
		if len(annotations) > 0 and self.feature_keys:
			for feat_key in self.feature_keys:
				for ann in annotations:
					if feat_key in ann:
						features.append(ann)
						
		return features
		
	def get_feature_vec(self, w_index, doc):
		features = self.get_features(w_index, doc)
		vec = torch.zeros(len(self.findex))
		for f in features:
			if f in self.findex:
				findex = self.findex[f]
				vec[findex] = 1.0
		if self.gpu:
			vec = vec.cuda()
		return autograd.Variable(vec, requires_grad=False)
		

	def get_tif_vec(self, w_index, doc):
		span = doc.spans[w_index]
		if span in doc.reverse_span_annotations:
			k = [tif for tif in doc.reverse_span_annotations[span] if tif[:3]=='TIF']
			#print(k)
			if len(k) >0:
				return self.tif_vecs[k[0]]
		return self.tif_vecs['TIF-UNKNOWN']
		
	def set_train_mode(self):
		for component in self.nn.values():
			if hasattr(component, 'train'):
				component.train()

	def set_eval_mode(self):
		for component in self.nn.values():
			if hasattr(component, 'eval'):
				component.eval()
	
	def __init__(self, model_dir='tml_model', data=[], margin=0.01, dmin=0.1, pemb_size=20, wemb_size=25, cemb_size=10, rnn_size=50, crnn_size=20, lr=0.001, gpu=True, relations=['BEFORE', 'AFTER', 'INCLUDES', 'IS_INCLUDED','SIMULTANEOUS'], dropout=0.5, depth=1, unk_threshold=0, special_conflation=False, rnn_unit='LSTM', pos=False, optimizer='adam', loss_func='Ldce', subtasks=['sc','dc','sa','da'], word_vectors=None, fix_wembs=False, dct_start_fixed=True, dct_duration_fixed=False, rnn_bias=True, linear_bias=True, use_character_level_encoding=True,doc_normalization=True,blinding=False, feature_keys = None, deep_word_modeling=False, entity_sequence=False, absolute=False, pointwise_loss='hinge'):
		self.model_dir = model_dir
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		self.unk_token = '_unk_'
		self.feature_keys = feature_keys.split(',') if feature_keys else None
		self.windex, self.cindex, self.pindex, self.findex = self.setup_vocabularies(data, unk_threshold, special_conflation=special_conflation, entity_sequence=entity_sequence)
		print ('wvocab:', len(self.windex), 'cvocab:', len(self.cindex), 'pvocab:', len(self.pindex), 'fvocab:', len(self.findex), '( using pos:', bool(pos),', features:', self.feature_keys, ')')
		print('features:', self.findex.keys())
		self.train_margin, self.pred_margin = margin, margin
		
		self.dmin, self.rels_train, self.loss_func, self.pointwise_loss = dmin, relations, loss_func, pointwise_loss
		self.gpu, self.optimizer_type, self.lr = gpu, optimizer, lr
		self.special_conflation=special_conflation
		self.entity_sequence=entity_sequence
		self.absolute = absolute
		self.doc_normalization=doc_normalization
	
		# Some stats about layer sizes (for easy usage later on)
		self.pemb_size, self.wemb_size, self.crnn_size, self.cemb_size, self.rnn_size = (pemb_size if pos else 0), wemb_size,(crnn_size if use_character_level_encoding else 0), (cemb_size if use_character_level_encoding else 0), rnn_size
		self.pos, self.use_character_level_encoding, self.blinding, self.dropout, self.rnn_unit, self.deep_word_modeling = pos, use_character_level_encoding, blinding, dropout, rnn_unit, deep_word_modeling
		
		# --- Constructing Network Components
		self.nn, self.constants  = OrderedDict(), OrderedDict()
		self.contextual_subtasks, self.word_level_subtasks = ['sc','dc'], ['sa','da']
		
		# Set which subtasks should be used for prediction 
		self.active_subtasks = subtasks
		print('Active subtasks',self.active_subtasks)
		
		# optional dropout
		if self.dropout:
			self.nn['dropout*'] = nn.Dropout(self.dropout)
		
		# Single parameters (or constants)
		self.nn['s_dct*'] = autograd.Variable(torch.zeros(1), requires_grad=True)
		self.nn['d_dct*'] = autograd.Variable(torch.ones(1), requires_grad=True)
		self.constants['ZERO'] = autograd.Variable(torch.FloatTensor([0]),requires_grad=False)
		
		# Word representation modules
		if word_vectors:
			wv = read_word_vectors(word_vectors)
		for subtask in self.contextual_subtasks + self.word_level_subtasks:
			if word_vectors:
				self.windex, self.nn['wembs_'+subtask], self.wemb_size = self.set_word_embeddings(wv)
			else:
				self.nn['wembs_'+subtask] = nn.Embedding(len(self.windex), self.wemb_size)
			if pos:
				self.nn['pembs_'+subtask] = nn.Embedding(len(self.pindex), self.pemb_size)
			if use_character_level_encoding:
				self.nn['cembs_'+subtask] = nn.Embedding(len(self.cindex), self.cemb_size)
				self.nn['crnn_'+subtask] = nn.LSTM(self.cemb_size, self.crnn_size, bidirectional=False, num_layers=depth, bias=rnn_bias)
		
		self.word_repr_size = self.pemb_size + self.wemb_size + self.crnn_size + (len(self.findex) if self.feature_keys else 0)
		if deep_word_modeling:
			for subtask in self.contextual_subtasks + self.word_level_subtasks:
				self.nn['wff_'+subtask] = nn.Linear(self.word_repr_size, deep_word_modeling)
			self.word_repr_size = deep_word_modeling
		# Contextual modules
		for subtask in self.contextual_subtasks:
			if self.rnn_unit == 'LSTM':
				self.nn['wrnn_'+subtask] = nn.LSTM(self.word_repr_size, self.rnn_size, bidirectional=True, num_layers=depth, bias=rnn_bias)
			elif self.rnn_unit == 'Att':
				self.nn['wrnn_'+subtask] = MultiHeadAttention(n_head=2, d_model=self.word_repr_size, d_k=10, d_v=10)
				self.nn['out_'+subtask] = nn.Linear(self.word_repr_size, 1, bias=linear_bias)
			
		# Non-contextual modules:
		self.out_repr_size_d = 0 + (1 if 'dp' in self.active_subtasks else 0) + (1 if 'sp' in self.active_subtasks else 0) + (2*self.rnn_size if 'dc' in self.active_subtasks else 0) + (self.word_repr_size if 'da' in self.active_subtasks else 0) 
		self.out_repr_size_s = 0 + (1 if 'dp' in self.active_subtasks else 0) + (1 if 'sp' in self.active_subtasks else 0) + (2*self.rnn_size if 'sc' in self.active_subtasks else 0) + (self.word_repr_size if 'sa' in self.active_subtasks else 0) 

		
		self.nn['out_s'] = nn.Linear(self.out_repr_size_d, 1, bias=linear_bias)
		self.nn['out_d'] = nn.Linear(self.out_repr_size_s, 1, bias=linear_bias)

		# Easy access to subparts of the net by subtask, to easily free or fix parameters
		self.nn_by_subtask = {subtask:{name:component for (name,component) in self.nn.items() if subtask in name or '*' in name} for subtask in self.contextual_subtasks+self.word_level_subtasks}
		for subtask in self.nn_by_subtask:
			self.nn_by_subtask[subtask]['out_s']=self.nn['out_s'] 
			self.nn_by_subtask[subtask]['out_d']=self.nn['out_d'] 
			
		# Set all components to trainable by default except checking the DCT start and duration
		self.tied_components = set()
		
		if dct_start_fixed:
			self.fix_component_by_name('s_dct*')
		if dct_duration_fixed:
			self.fix_component_by_name('d_dct*')

		self.reset_optimizer()
		print('Full model parameters:', sum([np.prod(list(par.size())) for par in self.get_trainable_params()]))
		print('Word representation size:',self.word_repr_size)				
		print ('Dims - wemb:',self.wemb_size, '- pemb:',self.pemb_size, '- cemb:',self.cemb_size, '- wrnn:', self.rnn_size, '- crnn:', self.crnn_size)
		print ('Relations:', relations)
		
		
		if self.gpu:
			self.move_to_gpu()
		

	def index_w(self, w):
		return self.windex[w] if w in self.windex else self.windex[self.unk_token]

	def index_p(self, p):
		return self.pindex[p] if p in self.pindex else self.pindex[self.unk_token]

	def index_c(self, c):
		return self.cindex[c] if c in self.cindex else self.cindex[self.unk_token]

	def get_e_vec(self, e):
		return self.e_vecs[e] if e in self.e_vecs else self.e_vecs[self.unk_token]

	def encode_char(self,c, subtask):
		return self.nn['cembs_'+subtask](self.index_c(c))
	
	def conflate_digits(self, w):
		return re.sub('\d', '5', w)

	def set_word_embeddings(self, wv):
		print('setting word embeddings')
		wv_vocab = [w for w in wv.vocab.keys() if (not ('_' in w) or w=='_') and w in self.windex]	# ! only words that overlap are initialized (so no bigger vocab)!	
		new_windex, wemb_size = {w:i for i,w in enumerate(wv_vocab + [self.unk_token])}, wv.vector_size
		wembs = nn.Embedding(len(new_windex), wemb_size)
		emb_matrix = np.zeros([len(new_windex), wemb_size], dtype=float)
		for w in new_windex:
			if w in wv:				
				emb_matrix[new_windex[w]] = wv[w]
		emb_tensor = torch.from_numpy(emb_matrix).float()
		wembs.weight.data = emb_tensor.view(len(new_windex), wemb_size)
		new_windex = {w:autograd.Variable(torch.from_numpy(np.array([i]))) for w,i in new_windex.items()}
		print ('vocab size:', len(wv_vocab))
		return new_windex, wembs, wemb_size
		
	def encode_word_for_subtask(self, w_index, doc, subtask):
		if self.entity_sequence:
			token_str = doc.span_to_tokens(doc.entity_spans[w_index])[-1]
		else:
			token_str = doc.tokens[w_index]	
		
		if self.blinding == 1 and subtask in ['dc','sc'	] and doc.entities[w_index]!='O':
			token_str = self.unk_token
		elif self.blinding == 2 and subtask in ['dc','sc']:
			token_str = self.unk_token

		# Getting the word embedding
		if self.special_conflation:
			word_encoding = self.nn['wembs_'+subtask](self.index_w(self.conflate_digits(token_str)))
		else:
			word_encoding = self.nn['wembs_'+subtask](self.index_w(token_str))

		# Adding Character RNN encoding
		if self.use_character_level_encoding:
			# Constructing sequence of char-embeddings
			cembs_lr = torch.stack([self.encode_char(c, subtask) for c in token_str])		
			
			# Running Char-RNN	
			cencoding_lr, _ = self.nn['crnn_'+subtask](cembs_lr)
		
			# Concatenating the word embedding and last Char-RNN output 
			word_encoding = torch.cat([word_encoding,cencoding_lr[-1]], dim=1)

		# Adding POS
		if self.pos:
			pemb = self.nn['pembs_'+subtask](self.index_p(doc.pos[w_index]))
			word_encoding = torch.cat([word_encoding, pemb], dim=1)

		# Adding Entity encoding (EVENT, TIMEX3, or NONE)
			
		if self.feature_keys:
			feat_vec = self.get_feature_vec(w_index, doc).view(1,-1)
			
			word_encoding = torch.cat([word_encoding, feat_vec], dim=1)
		
		if self.deep_word_modeling:
			word_encoding = torch.tanh(word_encoding)
			word_encoding = self.nn['wff_'+subtask](word_encoding)
			
		
		# Add dropout
		if self.dropout:
			word_encoding = self.nn['dropout*'](word_encoding)
		
		return word_encoding


	
	def encode_tokens_for_subtask(self, doc, subtask):
		# construct word representations
		if self.entity_sequence:
			word_encoding = torch.stack([self.encode_word_for_subtask(e_index,doc, subtask) for e_index in range(len(doc.entity_spans))])		
		else:
			word_encoding = torch.stack([self.encode_word_for_subtask(w_index,doc, subtask) for w_index in range(len(doc.tokens))])
  
		# For contextual subtasks apply the corresponding word-level RNN 
		if subtask in self.contextual_subtasks:

			if self.rnn_unit in ['LSTM', 'GRU','RNN']:
				word_encoding, _ = self.nn['wrnn_'+subtask](word_encoding) 
				
			if self.rnn_unit in ['Att']:
				word_encoding, enc_slf_attn = self.nn['wrnn_'+subtask](word_encoding, word_encoding, word_encoding) 
				
				
				
			# Add dropout (dropout is already appliedon word representation level as well)
			if self.dropout:
				word_encoding = self.nn['dropout*'](word_encoding)	
				
		return word_encoding


	def pred_subtask(self, token_index, doc, encoded_text, subtask):
		token_representation = encoded_text[subtask][token_index]
		return self.nn['out_'+subtask](token_representation)
		

	def encode_tokens(self, doc, entity_spans=None, measure_speed=False):
		if measure_speed:
			t0 = time.time()
		entity_spans = entity_spans if entity_spans else doc.entity_spans
		encodings = {}
		sp,dp = 0,0
		for subtask in self.active_subtasks:
			if not subtask in set(['dp','sp']):
				encodings[subtask] = self.encode_tokens_for_subtask(doc, subtask)

		encodings['s'], encodings['d'] = {},{}
		
		# span (0,0) corresponds to the document-creation-time		
		s, d = self.nn['s_dct*'].view(1,1), self.clamp(self.nn['d_dct*'].view(1,1), self.dmin)#.clamp(self.dmin)
		encodings['s'][(0,0)], encodings['d'][(0,0)] = s, d
		sp,dp = s, d
		
		for span in entity_spans:
		
			# Get the token index corresponding to the span
			token_ix = doc.entity_indices[span] if self.entity_sequence else  doc.span_to_tokens(span,token_index=True)[-1]
			
			tok_rs, tok_rd = None,None
			
			if 'sa' in self.active_subtasks:
				tok_rs = encodings['sa'][token_ix]
		
			if 'da' in self.active_subtasks:
				tok_rd = encodings['da'][token_ix]

			if 'sc' in self.active_subtasks:
				tok_rs = torch.cat([tok_rs, encodings['sc'][token_ix]], dim=1) if tok_rs is not None else encodings['sc'][token_ix]
			
			if 'dc' in self.active_subtasks:
				tok_rd = torch.cat([tok_rd, encodings['dc'][token_ix]], dim=1) if tok_rd is not None else encodings['dc'][token_ix]


			if 'sp' in self.active_subtasks:
				tok_rs = torch.cat([tok_rs, sp], dim=1)
				tok_rd = torch.cat([tok_rd, sp], dim=1)


			if 'dp' in self.active_subtasks:
				tok_rs = torch.cat([tok_rs, dp], dim=1)
				tok_rd = torch.cat([tok_rd, dp], dim=1)

	
			s, d = self.nn['out_s'](tok_rs), self.clamp(self.nn['out_d'](tok_rd), self.dmin)
			
				
			encodings['s'][span] = s
			encodings['d'][span] = d
			sp,dp = s, d
		
		if measure_speed:
			print(doc.id, 'enc t:',time.time()-t0,'s', 'words:', len(doc.tokens),'w/s:', float(len(doc.tokens)) / (time.time()-t0))
		return encodings		

	def clamp(self, tensor, min_value):
		return torch.log(1.0 + torch.exp(tensor)) + min_value
		
	def pred_starttime(self, span, doc, encoded_text):
		return encoded_text['s'][span]
			
	def pred_duration(self, span, doc, encoded_text):
		return encoded_text['d'][span]

	def pointwise_loss_before(self, x, y, train_mode=False):	# X < Y, interpreted as: max(X + m - Y, 0)
		margin_t = self.train_margin if train_mode else self.pred_margin

		
		if self.pointwise_loss == 'hinge':
			loss = torch.max(torch.stack([x[0] + margin_t - y[0], self.constants['ZERO']]))
		elif self.pointwise_loss == 'log':
			loss = torch.log(1 + torch.exp(x[0] - y[0] + margin_t))
		elif self.pointwise_loss == 'exp':
			loss = torch.exp(x[0] - y[0] + margin_t)
			
		return loss.view(1)
		
	def pointwise_loss_equal(self, x, y, train_mode=False):
		# |x-y| < margin   --> max(|x-y| - self.loss_margin , 0)
		margin_t = self.train_margin if train_mode else self.pred_margin

		if self.pointwise_loss == 'hinge':
			loss = torch.max(torch.stack([torch.abs(x[0] - y[0]) - margin_t, self.constants['ZERO']]))
		elif self.pointwise_loss == 'log':
			loss = torch.log(1 + torch.exp(torch.abs(x[0] - y[0]) - margin_t))
		elif self.pointwise_loss == 'exp':
			loss = torch.exp(torch.abs(x[0] - y[0]) - margin_t)
			
		return loss.view(1)
	
	def get_Lt(self, rel, s1, d1, s2, d2, train_mode=False):
		
		e1 = s1 + d1
		e2 = s2 + d2

		
		if rel == 'IS_INCLUDED':
			loss = self.pointwise_loss_before(s2, s1, train_mode) + self.pointwise_loss_before(e1, e2, train_mode) # + self.pointwise_loss_before(d1,d2)
		elif rel =='INCLUDES':
			loss = self.pointwise_loss_before(s1, s2, train_mode) + self.pointwise_loss_before(e2, e1, train_mode) # + self.pointwise_loss_before(d2,d1)
		elif rel == 'BEFORE':
			loss = self.pointwise_loss_before(e1, s2, train_mode) 
		elif rel == 'AFTER':
			loss = self.pointwise_loss_before(e2, s1, train_mode)
		elif rel == 'SIMULTANEOUS':
			loss = self.pointwise_loss_equal(s1, s2, train_mode) + self.pointwise_loss_equal(e1, e2, train_mode) # + self.pointwise_loss_equal(d1,d2)
		elif rel == 'BEGINS':
			loss = self.pointwise_loss_equal(s1, s2, train_mode) + self.pointwise_loss_before(e1, e2, train_mode)
		elif rel == 'BEGUN_BY':
			loss = self.pointwise_loss_equal(s2, s1, train_mode) + self.pointwise_loss_before(e2, e1, train_mode)
		elif rel  == 'ENDS':
			loss = self.pointwise_loss_before(s2, s1, train_mode) + self.pointwise_loss_equal(e1, e2, train_mode)
		elif rel  == 'ENDED_BY':
			loss = self.pointwise_loss_before(s1, s2, train_mode) + self.pointwise_loss_equal(e2, e1, train_mode)
		elif rel == 'IBEFORE':
			loss = self.pointwise_loss_equal(e1, s2, train_mode)
		elif rel == 'IAFTER':
			loss = self.pointwise_loss_equal(e2, s1, train_mode)
		else:
			print('ERROR: no loss for relation:', rel)

		#print(rel, loss, s1, e1, s2, e2)
		return loss	

	def get_Lr(self, rel, s1, d1, s2, d2, all_relations, train_mode=False):




		if self.loss_func == 'Lt':
			return self.get_Lt(rel, s1, d1, s2, d2, train_mode)

		elif self.loss_func == 'Ldh': # the timeline loss of the true label should be lower than that of all false/other labels
			gt_loss = self.get_Lt(rel, s1, d1, s2, d2, train_mode)
			loss = 0.0
			for other_rel in all_relations:
				if other_rel != rel:
					loss += torch.max(torch.stack([gt_loss - self.get_Lt(other_rel, s1, d1, s2, d2, train_mode) + self.dmin, self.constants['ZERO']]))
			return loss


		elif self.loss_func == 'Ldcem':
			# Uses standard normalization instead of softmax 	
			f = lambda x: -x
			score_per_relation = torch.stack([f(self.get_Lt(rel, s1, d1, s2, d2, train_mode))] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			
			lifted_scores = score_per_relation + (0 - torch.min(score_per_relation))
			minmaxnorm = lambda x: x / torch.sum(x)

			mm1 = minmaxnorm(lifted_scores)
			return 1 - mm1[0]
			
		elif self.loss_func == 'Ldcemt':
			# Uses standard normalization instead of softmax and use tanh to flatten low scores (and prevent forever pushing away from unlikely relations, causing the time-line to move always during learning)
			f = lambda x: torch.tanh(-x)
			score_per_relation = torch.stack([f(self.get_Lt(rel, s1, d1, s2, d2, train_mode))] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			
			lifted_scores = score_per_relation + (0 - torch.min(score_per_relation))
			minmaxnorm = lambda x: x / torch.sum(x)

			mm1 = minmaxnorm(lifted_scores)
			return 1 - mm1[0]	

		
		elif self.loss_func == 'Ldce':
			
			f = lambda x: -x 
			new_score = torch.stack([f(self.get_Lt(rel, s1, d1, s2, d2, train_mode))] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			score_per_relation = new_score
			
			ref_vector = autograd.Variable(torch.LongTensor([0]), requires_grad=False)
			if self.gpu:
				ref_vector = ref_vector.cuda()	
	
			cross_entropy = torch.nn.CrossEntropyLoss()
			return cross_entropy(score_per_relation.t(), ref_vector)
			
		elif self.loss_func in ['Lt+Ldh','Ldh+Lt']:
			gt_loss = self.get_Lt(rel, s1, d1, s2, d2, train_mode)
			loss = 0.0
			for other_rel in all_relations:
				if other_rel != rel:
					loss += torch.max(torch.stack([gt_loss - self.get_Lt(other_rel, s1, d1, s2, d2, train_mode) + self.dmin, self.constants['ZERO']]))
			return loss + gt_loss

		elif self.loss_func in ['Lt+Ldce','Ldce+Lt']:
			f = lambda x: -x 
			gt_loss = self.get_Lt(rel, s1, d1, s2, d2, train_mode)
			new_score = torch.stack([f(gt_loss)] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			score_per_relation = new_score
			
			ref_vector = autograd.Variable(torch.LongTensor([0]), requires_grad=False)
			if self.gpu:
				ref_vector = ref_vector.cuda()	
	
			cross_entropy = torch.nn.CrossEntropyLoss()
			return cross_entropy(score_per_relation.t(), ref_vector) + gt_loss

		elif self.loss_func in ['Ldh+Ldce','Ldce+Ldh']:
			gt_loss = self.get_Lt(rel, s1, d1, s2, d2, train_mode)
			f = lambda x: -x
			loss = 0.0
			for other_rel in all_relations:
				if other_rel != rel:
					loss += torch.max(torch.stack([gt_loss - self.get_Lt(other_rel, s1, d1, s2, d2, train_mode) + self.dmin, self.constants['ZERO']]))
			
			new_score = torch.stack([f(gt_loss)] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			score_per_relation = new_score
			ref_vector = autograd.Variable(torch.LongTensor([0]), requires_grad=False)
			if self.gpu:
				ref_vector = ref_vector.cuda()	
	
			cross_entropy = torch.nn.CrossEntropyLoss()
			loss += cross_entropy(score_per_relation.t(), ref_vector)			
			return loss			


		elif self.loss_func == 'L*':
			gt_loss = self.get_Lt(rel, s1, d1, s2, d2, train_mode)
			f = lambda x: -x
			loss = 0.0
			for other_rel in all_relations:
				if other_rel != rel:
					loss += torch.max(torch.stack([gt_loss - self.get_Lt(other_rel, s1, d1, s2, d2, train_mode) + self.dmin, self.constants['ZERO']]))
			
			new_score = torch.stack([f(gt_loss)] + [f(self.get_Lt(r, s1, d1, s2, d2, train_mode)) for r in all_relations if not r==rel])
			score_per_relation = new_score
			ref_vector = autograd.Variable(torch.LongTensor([0]), requires_grad=False)
			if self.gpu:
				ref_vector = ref_vector.cuda()	
	
			cross_entropy = torch.nn.CrossEntropyLoss()
			loss += cross_entropy(score_per_relation.t(), ref_vector)	

			loss += self.get_Lt(rel, s1, d1, s2, d2, train_mode)[0]
			return loss 

		
	def train(self, data, num_epochs=5, max_docs=None, viz_inbetween=False, verbose=0,save_checkpoints=None, eval_on=None, batch_size=32, temporal_awareness_ref_dir=None, clip=1.0, pred_relations=None, patience=100, loss_func=None, pointwise_loss=None,tune_margin=1, checkpoint_interval=1000,timex3_dur_loss=False, reset_optimizer=None):
		training_start_time = time.time()
		print('Fixed components:', self.tied_components)
		print('Trainable parameters:', sum([np.prod(list(par.size())) for par in self.get_trainable_params()]))				

		print ('epochs:', num_epochs, 'dropout:', self.dropout, 'batch_size:', batch_size)
		print('checkpoints:', save_checkpoints)
		torch.backends.cudnn.benchmark = True
		self.reset_optimizer()
		
		if loss_func:
			self.loss_func = loss_func
		if pointwise_loss:
			self.pointwise_loss=pointwise_loss
		print('Lr loss func:', self.loss_func)
		print('Lp loss func:',self.pointwise_loss)
			
		if max_docs:
			data = data[:max_docs]
		
		# Taking subsection from training to calculate training accuracy
		train_err_subset = data[:max(int(len(data)*0.05),5)]
		pred_relations = pred_relations if pred_relations else self.rels_train
		
		if save_checkpoints:
			checkpoint_dir = self.model_dir + '/checkpoints/'
			os.makedirs(checkpoint_dir)
		if eval_on:
			error_dir_conf = self.model_dir + '/errors/confusion/'
			error_dir_entities = self.model_dir + '/errors/entities/'
			os.makedirs(error_dir_conf)
			os.makedirs(error_dir_entities)
			dev_metrics, F1_TA, P_TA, R_TA = evaluate_timelinemodel(self, eval_on, pred_relations,temporal_awareness_ref_dir=temporal_awareness_ref_dir,all_pairs=True)
			train_metrics, _, _, _ = evaluate_timelinemodel(self, train_err_subset, pred_relations, all_pairs=True, entity_error_analysis_file_path=error_dir_entities+'/train_0.txt')

			save_confusion_matrix_from_metrics(train_metrics, error_dir_conf + '/train_0.html')
			save_confusion_matrix_from_metrics(dev_metrics, error_dir_conf + '/dev_0.html')
			# saving initial evaluation (before training)
			best_eval_acc = get_acc_from_confusion_matrix(dev_metrics)
			epoch_stats = {'loss':[None], 'eval_acc':[get_acc_from_confusion_matrix(dev_metrics)], 'train_acc':[get_acc_from_confusion_matrix(train_metrics)]}
			
			if temporal_awareness_ref_dir:
				epoch_stats['F1_TA'], epoch_stats['P_TA'], epoch_stats['R_TA'] = [F1_TA], [P_TA], [R_TA]
		else:
			best_eval_acc = 0,0	
			
		if viz_inbetween:
			viz_dir = self.model_dir + '/viz/'
			os.makedirs(viz_dir)
			viz_doc = data[0]
			self.pred_viz(viz_doc, path=viz_dir + '/timeline0.html')
		

		num_examples_seen, num_examples_seen_prev_chkpt = 0, 0
		batch_id = 0
		e = 0
		chkpt_id,best_chkpt = 0,0
		while (e < num_epochs + 1) and (chkpt_id - best_chkpt <= patience):
			e+=1
			# ------------------------------------- start of epoch ------------------------
			
			 # set network to training mode (for dropout)

			streaming_avg_loss = []
			start_time = time.time()
			batches = []
			num_batches_per_doc = {}
			for doc_id,doc in enumerate(data):
				c_rels = [(r, p) for (r,ps) in doc.span_pair_annotations.items() for p in ps if r in self.rels_train]
				random.shuffle(c_rels)	
				num_batches = int(len(c_rels)/batch_size) + 1
				num_batches_per_doc[doc_id] = num_batches
				batch_indices = range(num_batches)
				for batch_i in batch_indices:
					batch = c_rels[batch_i*batch_size:(batch_i+1)*batch_size]
					batches.append((doc_id,batch))
			
			random.shuffle(batches)
			print ('\n===== Epoch', e, '(',(len(data)),' docs,',len(batches),'batches ) =====\n')			
			
			self.set_train_mode()
			for doc_id, batch in batches:	
				if chkpt_id - best_chkpt > patience:
					print('no more patience...')
					break	
				
				if reset_optimizer and len(streaming_avg_loss) % reset_optimizer: # reset optimizer every X iterations
					self.reset_optimizer()
					
				doc, batch_start_time, batch_id, num_examples_seen = data[doc_id], time.time(), batch_id + 1, num_examples_seen + len(batch)
				loss, predicted_spans = 0.0, {}
				self.optimizer.zero_grad()
				encoded_text = self.encode_tokens(doc)
				
				
				# Make span predictions
				for rel, (span_a1, span_a2) in batch:
					if not span_a1 in predicted_spans:
						predicted_spans[span_a1] = self.pred_span(doc, span_a1, encoded_text, convert_to_floats=False)
					if not span_a2 in predicted_spans:
						predicted_spans[span_a2] = self.pred_span(doc, span_a2, encoded_text, convert_to_floats=False)		
				
				# Calculate TLink Loss
				for rel, (span_a1, span_a2) in batch:
					s1, d1 = predicted_spans[span_a1]
					s2, d2 = predicted_spans[span_a2]
							
					Lr = self.get_Lr(rel, s1, d1, s2, d2, pred_relations, train_mode=True).view(1)
					loss += Lr



				if self.absolute:
					# Calculate Span Loss
					for span in predicted_spans:
						#print('--------------')
						#print(doc.span_to_string(span))
						anns = doc.reverse_span_annotations[span] if span in doc.reverse_span_annotations else []
						vs = [ann.split(':')[1] for ann in anns if ann.split(':')[0] == 'value']
						value = vs[0] if len(vs) > 0 else None
					
						if value:
							num_seconds = get_dur_from_value(value)
							if num_seconds:
								gt_duration = float(num_seconds) / 86400 # to number of days
								s, d = predicted_spans[span]
								
								#print('gt',num_seconds, gt_duration, d)
								Ldur = torch.abs(d - gt_duration).view(1)
								#print('Ldur>>', Ldur)
								loss += Ldur
							

				if self.doc_normalization:
					loss = loss / num_batches_per_doc[doc_id]
				loss_end_time = time.time()
				batch_loss = loss.cpu().data.numpy()[0] / len(batch) if type(loss) != float else 0
				
				if batch_loss > 0:						
					loss.backward()
					#self.print_gradient_by_name()
						
					if clip:
						for params in self.get_trainable_params():
							nn.utils.clip_grad_norm(params,clip)
					self.optimizer.step()
						
					
				streaming_avg_loss.append(batch_loss)
				print (batch_id, '/',len(batches),  doc.id, '\tbatch_loss:', round(batch_loss,5), 'streaming_avg_loss:',round(np.mean(streaming_avg_loss[-100:]),5),'\t t:', round(loss_end_time - batch_start_time,2),'backprop t:',round(time.time()-loss_end_time,2))
				
				if  num_examples_seen - num_examples_seen_prev_chkpt > checkpoint_interval : # After every 10.000 examples evaluate the status quo
					chkpt_id += 1
					num_examples_seen_prev_chkpt = num_examples_seen
					self.set_eval_mode()
					if viz_inbetween:
						viz_start_time = time.time()
						self.pred_viz(viz_doc, path=viz_dir + '/timeline'+str(chkpt_id)+'.html')
						print ('viz t:',round(time.time() - viz_start_time, 2))
			
					avg_loss = np.mean(streaming_avg_loss[-100:])
					epoch_stats['loss'].append(avg_loss)
					print('\n-- checkpoint', chkpt_id, '--')
					print('> avg loss: [', avg_loss, '] examples seen:', num_examples_seen,'chkpt t:', round(time.time() - start_time,2))
					print('DCT\ts:', self.nn['s_dct*'].data.cpu().numpy(),'\td:',self.clamp(self.nn['d_dct*'], self.dmin).data.cpu().numpy())
				
					if eval_on:
						
						start_time_eval = time.time()
						print('eval rels:', pred_relations)
						original_margin = self.pred_margin
						m_range = set([max(original_margin+d,0) for d in np.arange(-0.15, 0.2, 0.05)]) if tune_margin == 2 else [original_margin] 
						best_m_acc, best_m = 0, original_margin
						for test_margin in m_range:
							self.pred_margin = test_margin
							dev_metrics, F1_TA, P_TA, R_TA = evaluate_timelinemodel(self, eval_on, pred_relations,temporal_awareness_ref_dir=temporal_awareness_ref_dir, all_pairs=True, entity_error_analysis_file_path=error_dir_entities + '/dev_' +str(chkpt_id) + '.txt')
							eval_acc=get_acc_from_confusion_matrix(dev_metrics)

						if tune_margin == 2:
							print('m:', round(test_margin, 3), 'eval_acc', round(eval_acc, 3))
					
						if eval_acc > best_m_acc:
							best_m, best_m_acc, best_eval_metric = test_margin, eval_acc, dev_metrics
						
						if temporal_awareness_ref_dir:
							best_F1_TA, best_P1_TA, best_R_TA = F1_TA, P_TA, R_TA
						self.pred_margin = best_m
				
						train_metrics, _, _, _ = evaluate_timelinemodel(self, train_err_subset, pred_relations, all_pairs=True, entity_error_analysis_file_path=error_dir_entities + '/train_' +str(chkpt_id) + '.txt')
						train_acc=get_acc_from_confusion_matrix(train_metrics)
						save_confusion_matrix_from_metrics(train_metrics, error_dir_conf + '/train_' + str(chkpt_id) + '-m'+ str(self.pred_margin) + '.html')
						save_confusion_matrix_from_metrics(best_eval_metric, error_dir_conf + '/dev_' + str(chkpt_id) + '-m'+ str(self.pred_margin) + '.html')
						epoch_stats['eval_acc'].append(eval_acc)
						epoch_stats['train_acc'].append(train_acc)
						if temporal_awareness_ref_dir:
							epoch_stats['F1_TA'].append(F1_TA)
							epoch_stats['P_TA'].append(P_TA)
							epoch_stats['R_TA'].append(R_TA)
							print ('M:',round(self.pred_margin,3), 'f1_ta', best_F1_TA,'p_ta', best_P1_TA, 'r_ta', best_R_TA, 'eval_acc:', round(best_m_acc, 3), 'train_acc:',round(train_acc, 3), 't:', round(time.time()-start_time_eval, 2))					
						else:
							print ('M:',round(self.pred_margin,3), '\teval_acc:', round(best_m_acc, 3), 'train_acc:',round(train_acc, 3), 't:', round(time.time()-start_time_eval, 2))

						if epoch_stats['eval_acc'][-1] >= best_eval_acc:
							print(epoch_stats['eval_acc'][-1],'>=', best_eval_acc)
							best_chkpt, best_eval_acc = chkpt_id, epoch_stats['eval_acc'][-1]
							if save_checkpoints:
								self.save_timelinemodel(checkpoint_dir + '/checkpoint_' + str(chkpt_id) + '.p')
							
						plot_data = [go.Scatter(x=np.array(range(num_epochs)), y=np.array(values), mode='lines+markers', name=key) for key,values in epoch_stats.items()]
						py.offline.plot(plot_data, filename=self.model_dir + '/train_stats.html', auto_open=False)

					print()
					self.set_train_mode()
		
		self.set_eval_mode()
		if save_checkpoints:
			best_checkpoint, best_score = best_chkpt, best_eval_acc
			print('>>> using best checkpoint:', best_checkpoint, 'with dev score', best_score)
			if best_checkpoint > 0:
				best_checkpoint_model = load_timelinemodel(checkpoint_dir + '/checkpoint_' + str(best_checkpoint) + '.p')
				print('setting checkpoint')
				self.__dict__.update(best_checkpoint_model.__dict__)
		
		
		if tune_margin:
			self.tune_pred_margin(data, pred_relations)
		
		
		self.save_timelinemodel(self.model_dir + '/model.p')
		print ('finished training t:',round(time.time()-training_start_time, 2))
		
	def pred_span(self, doc, span, encoded_text, convert_to_floats=True):
		start, duration = self.pred_starttime(span, doc, encoded_text), self.pred_duration(span, doc, encoded_text)

		if convert_to_floats:
			start, duration = float(start.cpu().data.numpy()[0,0]), float(duration.cpu().data.numpy()[0,0])
		return start, duration

	def start_duration_pair_to_relation(self, s1, d1, s2, d2, rels):
		# Returns the relation from rels that has the lowest Lt loss
		rel_losses = [(rel, self.get_Lt(rel, s1, d1, s2, d2).cpu().data.numpy()[0]) for rel in rels]
		return min(rel_losses, key=lambda x:x[1])[0]

	def pred_viz(self, doc, path='timeline.path'):

		 # https://plot.ly/python/gantt/
		encoded_text = self.encode_tokens(doc)
		events = {}
		
		dct_str = [label[6:] for label in doc.reverse_span_annotations[(0,0)] if 'value:' in label][0]
		dct_date_str = re.findall(r'\d\d\d\d-\d\d-\d\d', dct_str)[0]
		dct= datetime.datetime.strptime(dct_date_str, '%Y-%m-%d')
		
		for event_span in doc.span_annotations['EType:EVENT']:
			event_str  = doc.text[event_span[0]:event_span[1]]
			start, duration = self.pred_span(doc, event_span, encoded_text)
			events[event_str] = {'start_date':self.num_to_date(float(start),dct_date=dct), 'end_date':self.num_to_date(float(start + duration),dct_date=dct)}
		df_events = [dict(Task=event, Start=events[event]['start_date'], Finish=events[event]['end_date'], Resource='EVENT') for event in events]
		
		timex3s = {'DCT': {'start_date':self.num_to_date(float(0),dct_date=dct), 'end_date':self.num_to_date(float(0 + 1),dct_date=dct)}}
		for timex_span in doc.span_annotations['EType:TIMEX3']:
			timex3_str = doc.text[timex_span[0]:timex_span[1]]
			start, duration = self.pred_span(doc, timex_span, encoded_text)
			timex3s[timex3_str] = {'start_date':self.num_to_date(float(start),dct_date=dct), 'end_date':self.num_to_date(float(start + duration),dct_date=dct)}
		df_timex3 = [dict(Task=timex3, Start=timex3s[timex3]['start_date'], Finish=timex3s[timex3]['end_date'], Resource='TIMEX3') for timex3 in timex3s]

		colors = {'EVENT': 'rgb(0, 0, 255)', 'TIMEX3': 'rgb(0, 255, 100)' }

		fig = ff.create_gantt(sorted(df_events+df_timex3, key=lambda x: self.date_to_num(x['Start'])), title=doc.id, colors=colors, index_col='Resource',show_colorbar=True, group_tasks=True)
		py.offline.plot(fig, filename=path,auto_open=False)
	
	def predict_doc(self, doc, span_labels):
		self.set_eval_mode()
		encoded_text = self.encode_tokens(doc)
		for label in span_labels:
			for span in doc.span_annotations[label] + [(0,0)]:
				start, duration = self.pred_span(doc, span, encoded_text)
				st_lab, dur_lab = 'start:' + str(start), 'duration:' + str(duration)
				if not st_lab in doc.span_annotations:
					doc.span_annotations[st_lab] = []
				if not dur_lab in doc.span_annotations:
					doc.span_annotations[dur_lab] = []					
				doc.span_annotations[st_lab].append(span)
				doc.span_annotations[dur_lab].append(span)
		doc.reverse_span_annotations = reverse_dict_list(doc.span_annotations)
		return doc

	def classify_rels_in_doc(self, doc, rels, all_pairs=False):
		
		
		if all_pairs:
			pairs = set([pair for pair in doc.reverse_span_pair_annotations])
		else:
			pairs = set([pair for rel in rels if rel in doc.span_pair_annotations for pair in doc.span_pair_annotations[rel]])
		encoded_text = self.encode_tokens(doc)
		span_predictions = {}
		span_pair_predictions = {r:[] for r in rels}
		
		for a1,a2 in pairs:
			if not a1 in span_predictions:
				span_predictions[a1] = self.pred_span(doc, a1, encoded_text, convert_to_floats=False)
			if not a2 in span_predictions:
				span_predictions[a2] = self.pred_span(doc, a2, encoded_text, convert_to_floats=False)

			s1, d1 = span_predictions[a1]
			s2, d2 = span_predictions[a2]
			
			pred_rel = self.start_duration_pair_to_relation(s1, d1, s2, d2, rels)	
			span_pair_predictions[pred_rel].append((a1, a2))
		
		return span_pair_predictions,span_predictions
	
	def save_timelinemodel(self, path):
		print ('saving model', path)
		init_time = time.time()
		with open(path, 'wb') as f:
			pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
		print('saved t:',round(time.time()-init_time,2),'s')

	def parse_date(self, date):
		return dparser.parse(date)
		
	def date_to_num(self, date, dct_date=None):

		if not dct_date:
			dct_date = datetime.datetime(2017,10,12)
		return (date - dct_date).total_seconds()
	
	def num_to_date(self, num, dct_date=None):
		if not dct_date:
			dct_date = datetime.datetime(2017,10,12)
		return dct_date + datetime.timedelta(0, num)

	def tune_pred_margin(self, dataset, relations, margins=np.arange(0,1,0.1/3), max_docs=10):
		print('Tuning prediction margin')
		print('Training margin:', self.train_margin)
		tuning_dir = self.model_dir + '/tuning_m/'
		os.mkdir(tuning_dir)
		random.shuffle(dataset)
		max_acc, max_margin = 0, 0
		for m in margins:
			self.pred_margin = m
			metrics, F1, P, R = evaluate_timelinemodel(self, dataset[:max_docs], relations, all_pairs=True)
			acc = get_acc_from_confusion_matrix(metrics)
			save_confusion_matrix_from_metrics(metrics, tuning_dir + '/m'+ str(self.pred_margin) + '.html')

			if acc > max_acc:
				max_acc = acc
				max_margin = m
			print('m:',round(m,3),'\tacc:', acc)
		print('best margin:', max_margin)
		self.pred_margin = max_margin	

		
		

def load_timelinemodel(path):
	print ('loading model', path)
	with open(path, 'rb') as f:
		return pickle.load(f)


def read_word_vectors(path):
	print('reading word vectors:', path)
	try:
		wv = KeyedVectors.load_word2vec_format(path, binary=True)
	except:
		wv = KeyedVectors.load_word2vec_format(path, binary=False)
	return wv
	
def write_average_durs_and_starts(model, preds, file_path):
	word_to_s, word_to_d = {}, {}
	pred_dir = '/'.join(file_path.split('/')[:-1])
	if not os.path.exists(pred_dir):
		os.makedirs(pred_dir)

	for doc in preds:
		encoded_text = model.encode_tokens(doc)
		
		for espan in doc.span_annotations['EType:EVENT']:
			s, d = model.pred_span(doc, espan, encoded_text, convert_to_floats=True)
			tok_str = doc.span_to_string(espan)
			
			if not tok_str in word_to_s:
				word_to_s[tok_str],word_to_d[tok_str] = [],[]
			word_to_s[tok_str].append(s)
			word_to_d[tok_str].append(d)
	word_to_avg_s = sorted([(w,np.mean(values),np.var(values)) for w,values in word_to_s.items()], key=lambda x: x[1])
	word_to_avg_d = sorted([(w,np.mean(values),np.var(values)) for w,values in word_to_d.items()], key=lambda x: x[1], reverse=True)
	
	with open(file_path, 'w') as f:
		f.write('--- Start Times Ascending --- (WORD, START, FREQ)\n\n')
		for w,avg_s,var_s in word_to_avg_s:
			f.write(w + '\t' + str(round(avg_s,4)) + '\t' + str(round(var_s,4)) + '\t' + str(model.word_frequencies[w] if w in model.word_frequencies else '<UNK>') + '\n')
	
		f.write('\n\n--- Durations Descending --- (WORD, DURATION, FREQ)\n\n')
		for w,avg_d,var_d in word_to_avg_d:
			f.write(w + '\t' + str(round(avg_d,4)) + '\t' + str(round(var_d,4)) + '\t' + str(model.word_frequencies[w] if w in model.word_frequencies else '<UNK>') + '\n')
	
	
def evaluate_timelinemodel(model, docs, rel_labels, temporal_awareness_ref_dir=None, all_pairs=False, error_viz_dir=None, entity_error_analysis_file_path=None, write_average_durations_and_starts=False,print_sd_preds=False):
	preds, entity_errors_per_doc = [], []
	
	for doc in docs:
		#remove relations that you don't want to evaluate on
		for rel in doc.span_pair_annotations:
			if not rel in rel_labels:
				doc.span_pair_annotations[rel] = []
				
		# copy ref doc text etc
		pred = copy(doc)
		# remove relation annotations
		pred.span_pair_annotations = {}
		# classify relations using the model
		pairwise_labels, pointwise_preds = model.classify_rels_in_doc(doc, rel_labels,all_pairs=all_pairs)
		pred.update_annotations(span_pair_update=pairwise_labels)
		preds.append(pred)
		if print_sd_preds:
			if not os.path.exists(print_sd_preds):
				os.mkdir(print_sd_preds)
			with open(print_sd_preds + '/' + doc.id + '.txt', 'w') as f:
				preds_string = '\n'.join([str(s[0][0].cpu().data.numpy()) + '\t'+str(d[0][0].cpu().data.numpy()) + '\t' + str(span) +'\t'+ doc.span_to_string(span) for (span, (s,d)) in sorted(pointwise_preds.items(), key=lambda x: x[0][0])])
				preds_string = 'start\tduration\tspan\ttext\n' + preds_string
				f.write(preds_string)
			

	if error_viz_dir:
		viz_docs_rel_difference(docs, preds, error_viz_dir)

		
	# evaluate predictions	
	metrics, entity_errors_per_doc = get_eval_metrics_docs(docs, preds, rel_labels, entity_error_analysis_file_path, error_viz_dir)

	if entity_error_analysis_file_path:
		save_entity_error_analysis(docs, entity_errors_per_doc, entity_error_analysis_file_path)

	if write_average_durations_and_starts:
		write_average_durs_and_starts(model, preds, write_average_durations_and_starts)
				
		
	if temporal_awareness_ref_dir:
		#print('[temporal awareness evaluation subscripts]')
		# write preds to tmp folder
		tmp_pred_dir = model.model_dir + '/tmp_preds_'+str(len(docs))+'/'
		if not os.path.exists(tmp_pred_dir):
			os.mkdir(tmp_pred_dir)
		else:
			shutil.rmtree(tmp_pred_dir)
			os.mkdir(tmp_pred_dir)
		if not temporal_awareness_ref_dir[-1]=='/':
			temporal_awareness_ref_dir = temporal_awareness_ref_dir + '/'

		write_timebank_folder(preds, tmp_pred_dir, verbose=0)
		
		# 1. normalize temporal graphs
		norm_cmd = 'java -jar ./tempeval-3-tools/TimeML-Normalizer/TimeML-Normalizer.jar -a "'+temporal_awareness_ref_dir+';'+tmp_pred_dir+'"'
		norm_out_str = subprocess.check_output(norm_cmd, shell=True,stderr=subprocess.STDOUT)
		
		# 2. eval
		eval_cmd = 'python2.7 ./tempeval-3-tools/evaluation-relations/temporal_evaluation.py '+temporal_awareness_ref_dir[:-1]+'-normalized/'+' '+tmp_pred_dir[:-1]+'-normalized/ '+str(0) 
		
		eval_out_str = subprocess.check_output(eval_cmd, shell=True).decode("utf-8") 

		F1, P, R =   [float(x) for x in eval_out_str.split('\n')[3].split()]

		return metrics, F1, P, R
	else:
		return metrics, None, None, None



def get_eval_metrics_docs(docs, preds, rel_labels, entity_error_analysis_file_path, error_viz_dir):
	entity_errors_per_doc = []
	metrics = {rel:{rel:0 for rel in rel_labels} for rel in rel_labels}
	for i in range(len(preds)):
		# evaluate prediction
		if error_viz_dir:
			pred_metrics, metrics_per_span = get_selective_rel_metrics(docs[i], preds[i], rels=rel_labels, print_pairwise_errors=error_viz_dir + '/pairwise_errors_viz/')
		else:
			pred_metrics, metrics_per_span = get_selective_rel_metrics(docs[i], preds[i], rels=rel_labels)
			
		if entity_error_analysis_file_path:
			entity_errors_per_doc.append(metrics_per_span)
					
		# summing results for all documents
		for ref_rel in metrics:
			for pred_rel in metrics[ref_rel]:
				metrics[ref_rel][pred_rel] += pred_metrics[ref_rel][pred_rel]
	return metrics, entity_errors_per_doc

class TimelineFinder(TimelineModel): # TL2RTL Model
	
	def __init__(self, timeml_docs, dmin=0.025, rels_train=['BEFORE','AFTER','INCLUDES','IS_INCLUDED','SIMULTANEOUS'], rels_pred=['BEFORE','AFTER','INCLUDES','IS_INCLUDED','SIMULTANEOUS']):
		# Builds timelines from TimeML files
		self.dmin=dmin	
		self.constants = {}
		self.constants['ZERO'] = autograd.Variable(torch.FloatTensor([0]),requires_grad=False)
		self.entity_starts = {doc.id:{eid:autograd.Variable(torch.FloatTensor([[0]]),requires_grad=True) for eid in doc.get_span_labels_by_regex('ei\d+').union(doc.get_span_labels_by_regex('t\d+')) }for doc in timeml_docs}
		self.entity_durations = {doc.id:{eid:autograd.Variable(torch.FloatTensor([[self.dmin]]),requires_grad=True) for eid in doc.get_span_labels_by_regex('ei\d+').union(doc.get_span_labels_by_regex('t\d+')) }for doc in timeml_docs}
		self.rels_pred = rels_pred
		self.rels_train = rels_train
		self.gpu=False
		self.unk_token = '__unk__'
		self.feature_keys = None
		self.windex, self.cindex, self.pindex, self.findex = self.setup_vocabularies(timeml_docs, 0, special_conflation=0, entity_sequence=0)

		return

	def encode_tokens(self, doc, entity_spans=None):
		if not doc.id in self.entity_starts:
			print('ERROR:', doc.id, 'not found in timeline encoded documents')
			exit()
		encodings = {'s':{}, 'd':{}}	
		for eid in self.entity_starts[doc.id]:
			if not eid in doc.span_annotations:
				print('ERROR: eid not in document annotations:', eid, doc.get_span_labels_by_regex(eid[:2]+'.*'))
				exit()
			spans = doc.span_annotations[eid]
			if len(spans) > 1:
				print('!!!!!!!', doc.id, eid)
			span = spans[0]
			s, d = s, d = self.entity_starts[doc.id][eid], self.clamp(self.entity_durations[doc.id][eid], self.dmin)
		
		
			encodings['s'][span] = s
			encodings['d'][span] = d
		return encodings
	

	def train(self, timeml_docs, num_epochs):
		print('\n===== Building Timeline for each Document =====')
		# Starting to construct timelines
		for doc in timeml_docs:
			params = list(self.entity_starts[doc.id].values()) + list(self.entity_durations[doc.id].values())
			optimizer = torch.optim.Adam(params, lr=0.001)
			print(doc.id)
			for i in range(0,num_epochs):
				optimizer.zero_grad()
				loss = 0.0
				num_rels = 0
				for rel_type in self.rels_train:
					if rel_type in doc.span_pair_annotations:
						for sp_a1, sp_a2 in doc.span_pair_annotations[rel_type]:
							eid_a1 = [label for label in doc.reverse_span_annotations[sp_a1] if label in self.entity_starts[doc.id]][0]
							eid_a2 = [label for label in doc.reverse_span_annotations[sp_a2] if label in self.entity_starts[doc.id]][0]
							s1, d1 = self.entity_starts[doc.id][eid_a1], self.clamp(self.entity_durations[doc.id][eid_a1], min_value=self.dmin)
							s2, d2 = self.entity_starts[doc.id][eid_a2], self.clamp(self.entity_durations[doc.id][eid_a2], min_value=self.dmin)
							loss += self.get_Lr(rel_type, s1, d1, s2, d2, self.rels_pred, train_mode=True).view(1)
							num_rels += 1

				loss.backward()
				optimizer.step()
				if loss == 0.0:
					break
			print('loss', loss, 'after',i+1,'steps')
	
