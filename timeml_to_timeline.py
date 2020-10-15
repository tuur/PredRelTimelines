# -*- coding: utf-8 -*-
"""
Code from the paper "Temporal Information Extraction by Predicting Relative Time-lines" by Artuur Leeuwenberg & Marie-Francine Moens, In Proceedings of EMNLP, Brussels, Belgium, 2018.

Used to build relative time-lines from TimeML data (TL2RTL).
"""
import argparse, sys, os, shutil, torch, pickle
from time import gmtime, strftime
from lib.models import TimelineFinder, evaluate_timelinemodel
from lib.timeml import read_timebank_folder, simplify_relations, get_num_tlinks
from lib.evaluation import get_acc_from_confusion_matrix, save_confusion_matrix_from_metrics
from lib.data import Logger

parser = argparse.ArgumentParser(description='TimeML to Timeline')
parser.add_argument('-timeml', type=str, default=None,#"../data/experiments/TimebankDense/train/",
                    help='Folder containing TimeML data (.tml files)')
parser.add_argument('-test', type=str, default=None,#'../data/experiments/TimebankDense/dev/',
                    help='Folder containing test data (.tml files, with at least annotated entities)')
parser.add_argument('-relations', type=str, default='BEFORE,AFTER,IBEFORE,IAFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,BEGINS,BEGUN_BY,ENDS,ENDED_BY',
                    help='Relations used for training, default: BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS,BEGINS,BEGUN_BY,ENDS,ENDED_BY')
parser.add_argument('-pred_relations', type=str, default='BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS',
                    help='Relations to be predicted from the timeline, default: BEFORE,AFTER,INCLUDES,IS_INCLUDED,SIMULTANEOUS')	
parser.add_argument('-exp_id', type=str, default="0",
                    help='Experiment Identifier (default:0)')
parser.add_argument('-simplification', type=int, default=6,
                    help='Simplification of the relations, default:6 (following Ning et al., 2017)')
parser.add_argument('-optimizer', type=str, default='adam',
                    help='Specify the optimization method (adam, adadelta, sgd, rmsprop, amsgrad), default:adam')																				
parser.add_argument('-min_duration', type=float, default=0.1,
                    help='Minimum event duration, default:0.1')
parser.add_argument('-loss_func', type=str, default='Lt',
                    help='Loss function used, default:Lt')																				
parser.add_argument('-pointwise_loss', type=str, default='hinge',
                    help='Ranking loss function used for the pointwise loss functions, from hinge,log,exp, default: hinge')
parser.add_argument('-train_margin', type=float, default=0.025,
                    help='Margin used in the loss function, default: 0.025')
parser.add_argument('-load_model', type=str, default=None,
                    help='Load timelines')
parser.add_argument('-num_epochs', type=int, default=1000,
                    help='Number of training epochs, default:1000')
parser.add_argument('-pos', type=int, default=0,
                    help='Print POS in error analysis, default:0')
args = parser.parse_args()

dmin=0.1

exp_dir = 'out/'+args.exp_id
if os.path.exists(exp_dir):
	shutil.rmtree(exp_dir)
os.makedirs(exp_dir)
sys.stdout = Logger(stream=sys.stdout, file_name=exp_dir + '/log.log', log_prefix=str(args))
sys.stderr = sys.stdout
print('Today:',strftime("%Y-%m-%d %H:%M:%S", gmtime()))
print('PID:',os.getpid())
print('Experiment:',args.exp_id)


if args.timeml:
	timeml_docs = read_timebank_folder(args.timeml)
	# Read documents
	for doc in timeml_docs:
		simplify_relations(doc, simplification=args.simplification)

# get relation types to be used for timeline construction
relation_types = args.relations.split(',')
pred_relation_types = args.pred_relations.split(',')

# Build Timelinemodel (to access loss functions etc)
if args.load_model:
	with open(args.load_model, 'rb') as f:
		TLF = pickle.load(f)
		
else:
	TLF = TimelineFinder(timeml_docs, dmin=args.min_duration, rels_train=relation_types, rels_pred=pred_relation_types)
	TLF.loss_func = args.loss_func
	TLF.train_margin=args.train_margin
	TLF.pred_margin=args.train_margin
	TLF.pointwise_loss=args.pointwise_loss
	TLF.model_dir=exp_dir
	TLF.pred_rels=pred_relation_types


TLF.entity_durations = {docid.replace('sarcozy','sarkozy'):TLF.entity_durations[docid] for docid in TLF.entity_durations}
TLF.entity_starts = {docid.replace('sarcozy','sarkozy'):TLF.entity_starts[docid] for docid in TLF.entity_starts}



if args.timeml:	
	TLF.train(timeml_docs, args.num_epochs)
	
# Saving timelines	
with open(exp_dir + '/TLF.p', 'wb') as f:
	print('Saving model to', exp_dir + '/TLF.p')
	pickle.dump(TLF, f)


if args.timeml:
	# Evaluating on Train (should be close to 100% if (1) the TimeML relations are consistent and (2) the params are well optimized)
	print('\n--- Evaluating on Train--- ')
	train_metrics, F1, P, R = evaluate_timelinemodel(model=TLF, docs=timeml_docs,rel_labels=pred_relation_types, temporal_awareness_ref_dir=args.timeml, all_pairs=True)
	print ('TRAIN ACC:', round(get_acc_from_confusion_matrix(train_metrics), 3),'P_TA:', round(P, 3), 'R_TA:', round(R, 3), 'F1_TA:', round(F1, 3))
	save_confusion_matrix_from_metrics(train_metrics, exp_dir+'/train_confusion.html')

# Evaluating on test
if args.test:
	print('\n--- Evaluating on Test ---')
	test_docs = read_timebank_folder(args.test, pos=args.pos)
	TLF.feature_keys =  None
	TLF.unk_token='__unk__'
	TLF.setup_vocabularies(test_docs, 0, special_conflation=0, entity_sequence=0)
	print ('Documents:', len(test_docs))
	print ('TLINKS:', get_num_tlinks(test_docs), 'TOTAL:', sum(get_num_tlinks(test_docs).values()))
	

	print('read', exp_dir)
	test_metrics, F1, P, R = evaluate_timelinemodel(model=TLF, docs=test_docs,rel_labels=pred_relation_types, temporal_awareness_ref_dir=args.test, all_pairs=True, error_viz_dir=exp_dir+'/test/error_viz/', write_average_durations_and_starts=exp_dir+'/avg_predictions/avg_d&s.txt', entity_error_analysis_file_path=exp_dir + '/error_viz_test.txt', print_sd_preds=exp_dir+'/sd_preds/')
	print ('TEST ACC:', round(get_acc_from_confusion_matrix(test_metrics), 3),'P_TA:', round(P, 3), 'R_TA:', round(R, 3), 'F1_TA:', round(F1, 3))
	save_confusion_matrix_from_metrics(test_metrics, exp_dir+'/test_confusion.html', print_latex=True)
	

		






