# -*- coding: utf-8 -*-
"""
Code from the paper "Temporal Information Extraction by Predicting Relative Time-lines" by Artuur Leeuwenberg & Marie-Francine Moens, In Proceedings of EMNLP, Brussels, Belgium, 2018.

Used to train the S-TLM and C-TLM models from the paper.
"""

from __future__ import print_function
import time, argparse, sys, os, shutil
from lib.timeml import read_timebank_folder, simplify_relations,get_num_tlinks,extend_tlinks_with_timex3_values, add_TIF_features
from lib.models import TimelineModel, load_timelinemodel, evaluate_timelinemodel
#from lib.pairwisemodel import PairwiseModel
from lib.evaluation import get_acc_from_confusion_matrix, save_confusion_matrix_from_metrics
from lib.data import Logger
import torch, random
from time import gmtime, strftime

init_time = time.time()
# Timebank Dense 
#relations : BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS

# TempEval 3 Setting:
# relations: BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,BEGINS,BEGUN_BY,ENDS,ENDED_BY,IAFTER,IBEFORE


parser = argparse.ArgumentParser(description='Relative Time-line models')
parser.add_argument('-train', type=str, default=None,
                    help='Folder containing training data (.tml files)')
parser.add_argument('-dev', type=str, default=None,
                    help='Folder containing development data (.tml files)')
parser.add_argument('-test', type=str, default=None,
                    help='Folder containing test data (.tml files, with at least annotated entities)')
parser.add_argument('-relations', type=str, default='BEFORE,AFTER,IBEFORE,IAFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,BEGINS,BEGUN_BY,ENDS,ENDED_BY',
                    help='Relations used for training, default: BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,BEGINS,BEGUN_BY,ENDS,ENDED_BY')
parser.add_argument('-pred_relations', type=str, default='BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS',
                    help='Relations to be predicted, default: BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS')																				
parser.add_argument('-exp_id', type=str, default="0",
                    help='Experiment Identifier (default:0)')
parser.add_argument('-max_train_size', type=int, default=None,
                    help='Maximum on training set size, default: None')
parser.add_argument('-closure', type=int, default=0,
                    help='Use temporal transitive closure on relations, default: 0')
parser.add_argument('-max_test_size', type=int, default=None,
                    help='Maximum on training set size, default: None')
parser.add_argument('-num_epochs', type=int, default=1000,
                    help='Number of training epochs, default:1000')
parser.add_argument('-simplification', type=int, default=6,
                    help='Simplification of the relations, default:6 (following Ning et al., 2017)')
parser.add_argument('-wrnn_dim', type=int, default=25,
                    help='Number word-level RNN units, default:25')
parser.add_argument('-wrnn_depth', type=int, default=1,
                    help='Number RNN Layers, default:1')																				
parser.add_argument('-crnn_dim', type=int, default=10,
                    help='Number character-level RNN units, default: 10')
parser.add_argument('-wemb_dim', type=int, default=50,
                    help='Word-embedding dimension, default:50')
parser.add_argument('-cemb_dim', type=int, default=5,
                    help='Character-embedding dimension, default:5')
parser.add_argument('-pemb_dim', type=int, default=10,
                    help='POS-embedding dimension, default:10')
parser.add_argument('-conflate_digits', type=int, default=0,
                    help='Conflates digits to the same number as preprocessing (123->555), default:0') 
parser.add_argument('-special_conflation', type=int, default=0,
                    help='Conflates digits to the same number only on the word embedding level, but keeps original numbers for character-level (123->555), default:0') 
parser.add_argument('-unk', type=int, default=0,
                    help='Considers all words with a frequency below a certain UNK threshold as UNK. Default: 0')
parser.add_argument('-dropout', type=float, default=0.1,
                    help='Use dropout on the second last layer, default:0.1')
parser.add_argument('-batch_size', type=int, default=32,
                    help='Percentage of relations takes from each document per epoch, default:32')
parser.add_argument('-cuda', type=int, default=0,
                    help='Set cuda device, default:0')
parser.add_argument('-pos', type=int, default=1,
                    help='Use part-of-speech tags, default:1')
parser.add_argument('-gpu', type=int, default=1,
                    help='Use GPUs or not, default:1')
parser.add_argument('-clip', type=float, default=5.0,
                    help='Use gradient clipping, default: 5.0')
parser.add_argument('-load_model', type=str, default=None,
                    help='Loads a model from a pickled model file, default:None')
parser.add_argument('-train_margin', type=float, default=0.025,
                    help='Margin used in the loss function, default: 0.025')																			
parser.add_argument('-min_duration', type=float, default=0.1,
                    help='Minimum event duration, default:0.1')
parser.add_argument('-rnn_unit', type=str, default='LSTM',
                    help='Type of RNN that is used, GRU or LSTM, default: LSTM')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='Specify the optimization method (adam, adadelta, sgd, rmsprop, amsgrad), default:adam')
parser.add_argument('-subtasks', type=str, default='dc,sc',
                    help='Type subpredictions used to predict the final start and durations (from: da,sa,dc,sc,dp,sp), default: dc,sc')
parser.add_argument('-fix_subtasks', type=str, default='wembs_da,wembs_sa,wembs_sc,wembs_dc',
                    help='Fixes parameters of certain subtask networks, default:None')
parser.add_argument('-word_vectors', type=str, default='data/wordvectors/glove.6B.50d.w2vf.txt',
                    help='Use pre-trained word embeddings, providing a .bin vector file, default: data/wordvectors/glove.6B.50d.w2vf.txt')
parser.add_argument('-lowercase', type=int, default=1,
                    help='Lowercase the texts as preprocessing, default:1')
parser.add_argument('-loss_func', type=str, default='Ldce',
                    help='Loss function used, default:Ldce')																	
parser.add_argument('-checkpoints', type=int, default=1,
                    help='Save checkpoints during training, default: 1')
parser.add_argument('-fix_wembs', type=int, default=0,
                    help='Fix word embedding weights, default: 0')
parser.add_argument('-timex3_extension', type=int, default=0,
                    help='Extends the set of training TLINKS among timex3 instances, using their values, default:0')
parser.add_argument('-sim_extension', type=int, default=0,
                    help='Extends the set of training TLINKS by copying all incoming relations for events that occur SIMULTANEOUS (more TLINKS so slower training...), default:0')
parser.add_argument('-dct_duration_fixed', type=int, default=0,
                    help='DCT duration is fixed to 1 and not learnable, default:0')
parser.add_argument('-dct_start_fixed', type=int, default=1,
                    help='DCT start is fixed to 1 and not learnable, default:1')
parser.add_argument('-random_seed', type=int, default=1,
                    help='Random seed, default:1')
parser.add_argument('-tune_margin', type=int, default=1,
                    help='Tunes the margin for prediction (1: at prediction, 2: at prediction and training), default:2')
parser.add_argument('-set_pred_margin', type=float, default=None,
                    help='Resets the prediction margin of the model just before testing, default:None')
parser.add_argument('-patience', type=int, default=100,
                    help='Patience for early stopping, default:50')
parser.add_argument('-lr', type=float, default=0.001,
                    help='Learning rate, default:0.001 (default for Adam)')
parser.add_argument('-linear_bias', type=int, default=1,
                    help='Using a linear bias at the top layers, default:1')
parser.add_argument('-rnn_bias', type=int, default=1,
                    help='Using a biases for the RNNs at the top layers, default:1')
parser.add_argument('-character_level_encoding', type=int, default=0,
                    help='Use Character-level RNN in word encoding, default:0')																				
parser.add_argument('-deep_word_modeling', type=int, default=0,
                    help='Use a deep FFN to model word representations, default:0')
parser.add_argument('-doc_normalization', type=int, default=0,
                    help='Normalizes loss over the documents (each document counts equally), default:0')					
parser.add_argument('-blinding', type=int, default=0,
                    help='Blinds entities for context subtasks dc,sc, default:0')
parser.add_argument('-features', type=str, default="class,polarity,aspect,tense,EType,type,TIF",
                    help='Use temporal interval features from Do et al. (2012), from class,polarity,aspect,tense,EType,type,TIF,VPOS,PREP,VINDEX ,default: class,polarity,aspect,tense,EType,type,TIF')
parser.add_argument('-entity_sequence', type=int, default=0,
                    help='Use sequence of entities only instead of all words, default:0')
parser.add_argument('-checkpoint_interval', type=int, default=1000,
                    help='Interval by which the model is evaluated and a possible checkpoint is made (every X number of examples), default:1000')
parser.add_argument('-absolute', type=int, default=0,
                    help='Durations can be mapped directly to the number of days, default:0')
parser.add_argument('-event_filter', type=int, default=0,
                    help='Filters training documents when they have more than X events (60 or 80 for Ning et al., 2017), default: 0')
parser.add_argument('-max_sentence_distance', type=int, default=0,
                    help='Filters training TLinks by maximum sentence distance, default: 0')																				
parser.add_argument('-ning_closure', type=int, default=0,
                    help='Applies closure as by Ning et al. (2017), default: 0')																				
parser.add_argument('-pointwise_loss', type=str, default='hinge',
                    help='Ranking loss function used for the pointwise loss functions, from hinge,log,exp, (log works best) default: hinge')
parser.add_argument('-reset_optimizer', type=int, default=0,
                    help='Resets the optimizer every X iterations, default:0 ')																				
#parser.add_argument('-pairwise', type=int, default=0,
#                    help='Use pairwise relation classification mode, default:0 ')



args = parser.parse_args()

random.seed(args.random_seed)
exp_dir = 'out/'+args.exp_id
if os.path.exists(exp_dir):
	shutil.rmtree(exp_dir)
os.makedirs(exp_dir)
sys.stdout = Logger(stream=sys.stdout, file_name=exp_dir + '/log.log', log_prefix=str(args))
sys.stderr = sys.stdout
print('Today:',strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print('PID:',os.getpid())
print('Experiment:',args.exp_id)
print ('Available devices ', torch.cuda.device_count())
torch.cuda.set_device(args.cuda)
print ('Current cuda device ', torch.cuda.current_device())

if args.load_model:
	# ===============================
	print (10*'=', 'LOADING MODEL', 10*'=') # LOADING EXISTING MODEL ------------------
	# ===============================
	TLM = load_timelinemodel(args.load_model)
	TLM.active_subtasks = args.subtasks.split(',')
	args.pos = TLM.pos
	args.features=','.join(TLM.feature_keys)

if args.word_vectors and not os.path.isfile(args.word_vectors):
	print('WARNING: Word vectors', args.word_vectors, 'not found!!!')
	args.word_vectors = None

# ===============================
print (10*'=', 'READING DATASET', 10*'=') # READING DATASET ------------------
# ===============================

selected_labels = set(args.relations.split(','))
pred_relations = set(args.pred_relations.split(',')) if args.pred_relations else selected_labels

data = {'test':[], 'train':[], 'dev':[]}
if args.train:
	print ('--- TRAINING ---')
	data['train'] = read_timebank_folder(args.train, conflate_digits=args.conflate_digits, pos=args.pos, lowercase=args.lowercase)
if args.dev:
	print ('--- DEVELOPMENT ---')
	data['dev'] = read_timebank_folder(args.dev, conflate_digits=args.conflate_digits, pos=args.pos, lowercase=args.lowercase)
else:
	random.shuffle(data['train'])
	data['dev'] = data['train'][:int(len(data['train'])*0.09) + 1]
	data['train'] = data['train'][int(len(data['train'])*0.09) + 1:]
if args.test:
	print ('--- TESTING ---')
	data['test'] = read_timebank_folder(args.test, conflate_digits=args.conflate_digits, pos=args.pos, lowercase=args.lowercase)


# ===============================
print (10*'=', 'PREPROCESSING', 10*'=') # PREPROCESSING ------------------
# ===============================

if args.event_filter:
	data['train'] = [d for d in data['train'] if len(d.span_annotations['EType:EVENT']) > args.event_filter]	

if args.max_sentence_distance:
	for doc in data['train']:
		doc.filter_distant_pairs(args.max_sentence_distance)

if args.max_train_size:
	data['train'] = data['train'][:args.max_train_size] 
if args.max_test_size:
	data['dev'] = data['dev'][:args.max_test_size] 
	data['test'] = data['test'][:args.max_test_size]


print('found relations (in training) before simplification:', set([rel for doc in data['train'] for rel in doc.span_pair_annotations]))


if args.timex3_extension:
	for dataset in ['train']:
		for doc in data[dataset]:
			extend_tlinks_with_timex3_values(doc)

if 'TIF' in args.features:
	for dataset in ['train','dev','test']:
		for doc in data[dataset]:
			add_TIF_features(doc)

if ('VTOK' in args.features or 'VPOS' in args.features or 'VINDEX' in args.features) and args.pos:
	for dataset in ['train','dev','test']:
		for doc in data[dataset]:
			doc.add_VERB_features()

if 'PREP' in args.features and args.pos:
	for dataset in ['train','dev','test']:
		for doc in data[dataset]:
			doc.add_PREP_features()

if args.simplification:
	print('simplifying training relations')
	for dataset in ['train','dev']:
		for doc in data[dataset]:
			simplify_relations(doc, simplification=args.simplification)


print('conflated digits:', bool(args.conflate_digits))	
print('special conflated digits:', bool(args.special_conflation))		
print('found relations (in training):', set([rel for doc in data['train'] for rel in doc.span_pair_annotations]))
if args.closure:
	closure_relations = [r for r in ['BEFORE','AFTER','INCLUDES', 'IS_INCLUDED','SIMULTANEOUS'] if r in selected_labels]
	for doc in data['train']:
		for rel in closure_relations:
			doc.take_transitive_closure(rel)
			
elif args.ning_closure:
	for doc in data['train']:
		doc.ning_closure(args.ning_closure>1)
	
if args.sim_extension:
	for doc in data['train']:		
		doc.sim_rel_extension()	

print ('Documents:', {section:len(docs) for section,docs in data.items()})
print ('TLINKS:', {section:(get_num_tlinks(docs),{'TOTAL:':sum(get_num_tlinks(docs).values())}) for section,docs in data.items()})

if args.train:
	# ===============================
	print (10*'=', 'MAKING MODEL', 10*'=') # MAKING MODEL ------------------
	# ===============================
	
	if not args.load_model:
		vocab_data = data['train'] + data['dev'] + (data['test'] if args.test else []) # just used to select the vocabulary
		#if args.pairwise:
		#	TLM = PairwiseModel(model_dir=exp_dir,data=vocab_data, margin=args.train_margin, dmin=args.min_duration, relations=selected_labels, lr=args.lr, rnn_size=args.wrnn_dim, crnn_size=args.crnn_dim, wemb_size=args.wemb_dim, cemb_size=args.cemb_dim, pemb_size=args.pemb_dim, gpu=args.gpu, dropout=args.dropout, depth=args.wrnn_depth, unk_threshold=args.unk, special_conflation=args.special_conflation, rnn_unit=args.rnn_unit, pos=args.pos,optimizer=args.optimizer, loss_func=args.loss_func, feature_keys=args.features, subtasks=args.subtasks.split(','), word_vectors=args.word_vectors, fix_wembs=args.fix_wembs,dct_start_fixed=args.dct_start_fixed, dct_duration_fixed=args.dct_duration_fixed, linear_bias=args.linear_bias, rnn_bias=args.rnn_bias, use_character_level_encoding=args.character_level_encoding,doc_normalization=args.doc_normalization, blinding=args.blinding,deep_word_modeling=args.deep_word_modeling, entity_sequence=args.entity_sequence, absolute=args.absolute, pointwise_loss=args.pointwise_loss)
		#else:
		TLM = TimelineModel(model_dir=exp_dir,data=vocab_data, margin=args.train_margin, dmin=args.min_duration, relations=selected_labels, lr=args.lr, rnn_size=args.wrnn_dim, crnn_size=args.crnn_dim, wemb_size=args.wemb_dim, cemb_size=args.cemb_dim, pemb_size=args.pemb_dim, gpu=args.gpu, dropout=args.dropout, depth=args.wrnn_depth, unk_threshold=args.unk, special_conflation=args.special_conflation, rnn_unit=args.rnn_unit, pos=args.pos,optimizer=args.optimizer, loss_func=args.loss_func, feature_keys=args.features, subtasks=args.subtasks.split(','), word_vectors=args.word_vectors, fix_wembs=args.fix_wembs,dct_start_fixed=args.dct_start_fixed, dct_duration_fixed=args.dct_duration_fixed, linear_bias=args.linear_bias, rnn_bias=args.rnn_bias, use_character_level_encoding=args.character_level_encoding,doc_normalization=args.doc_normalization, blinding=args.blinding,deep_word_modeling=args.deep_word_modeling, entity_sequence=args.entity_sequence, absolute=args.absolute, pointwise_loss=args.pointwise_loss)
			
		TLM.pred_viz(data['dev'][0], path=exp_dir + '/timeline-dev0.html') # viz dev sample BEFORE training
	else:
		TLM.model_dir = exp_dir

	if args.fix_subtasks:
		for subtask in args.fix_subtasks.split(','):
			print('fixing', subtask)
			TLM.fix_component_by_name(subtask)
	
	print('eval_relations:',pred_relations)
	# ===============================
	print (10*'=', 'TRAINING', 10*'=') # TRAINING ------------------
	# ===============================
	

	TLM.train(data['train'], max_docs=args.max_train_size, num_epochs=args.num_epochs, viz_inbetween=False, verbose=0, save_checkpoints=args.checkpoints, eval_on=data['dev'], batch_size=args.batch_size,temporal_awareness_ref_dir=args.dev, clip=args.clip, pred_relations=pred_relations, loss_func=args.loss_func, tune_margin=args.tune_margin, patience=args.patience, checkpoint_interval=args.checkpoint_interval, reset_optimizer=args.reset_optimizer)


elif args.tune_margin and data['dev']:
	print('Tuning margin on dev')
	TLM.tune_pred_margin(data['dev'], pred_relations)
	TLM.save_timelinemodel(args.load_model[:-2] + '_tuned.p')

if args.set_pred_margin != None:
	print('Setting pred margin to', args.set_pred_margin)	
	TLM.pred_margin = args.set_pred_margin
	
# ===============================
print (10*'=', 'EVALUATING', 10*'=') # EVALUATING ------------------
# ===============================

if data['dev']:
	dev_metrics, F1, P, R = evaluate_timelinemodel(TLM, data['dev'], pred_relations, temporal_awareness_ref_dir=args.dev,all_pairs=True, error_viz_dir=TLM.model_dir+'/dev/error_viz/')
	if F1:
		print ('DEV ACC:', round(get_acc_from_confusion_matrix(dev_metrics), 3),'P_TA:', round(P, 3), 'R_TA:', round(R, 3), 'F1_TA:', round(F1, 3))
	else:
		print ('DEV ACC:', round(get_acc_from_confusion_matrix(dev_metrics), 3))
	TLM.pred_viz(data['dev'][0], path=exp_dir +'/timeline-dev1.html')  # viz dev sample AFTER training
	save_confusion_matrix_from_metrics(dev_metrics, exp_dir+'/dev_confusion.html')


if data['test']:
	test_metrics, F1, P, R = evaluate_timelinemodel(TLM, data['test'], pred_relations, temporal_awareness_ref_dir=args.test, all_pairs=True, error_viz_dir=TLM.model_dir+'/test/error_viz/', write_average_durations_and_starts=TLM.model_dir+'/avg_predictions/avg_d&s.txt',print_sd_preds=exp_dir+'/sd_preds/')
	if F1:
		print ('TEST ACC:', round(get_acc_from_confusion_matrix(test_metrics), 3),'P_TA:', round(P, 3), 'R_TA:', round(R, 3), 'F1_TA:', round(F1, 3),)
		print(round(P, 1), '&', round(R, 1), '&', round(F1, 1))
	else:
		print ('TEST ACC:', round(get_acc_from_confusion_matrix(test_metrics), 3))
	TLM.pred_viz(data['test'][0], path=exp_dir +'/timeline-test1.html')  # viz dev sample AFTER training
	save_confusion_matrix_from_metrics(test_metrics, exp_dir+'/test_confusion.html', print_latex=True)


# ===============================
print (10*'=', 'END T:', round(time.time()-init_time, 2)) # END ------------------
 # ===============================





