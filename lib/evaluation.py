from __future__ import print_function
import numpy as np
from lib.data import reverse_dict_list
import plotly as py
from plotly.graph_objs import Figure, Data
from copy import deepcopy
import os, shutil
# EVALUTION BASED ON SPAN AND SPAN PAIR PREDICTIONS


def get_acc_from_confusion_matrix(matrix):
	correct = 0
	total = 0
	for ref_rel in matrix:
		for pred_rel in matrix[ref_rel]:
			total += matrix[ref_rel][pred_rel]
			if ref_rel == pred_rel:
				correct += matrix[ref_rel][pred_rel]

	return correct / (total + 0.00000000001)

def save_confusion_matrix_from_metrics(matrix, file_path, print_latex=False):
	y_labels = list(matrix.keys())
	x_labels = list(reversed(y_labels)) + [l for l in list(matrix[list(matrix.keys())[0]].keys()) if not l in y_labels]
	z_values = [[matrix[y][x] for x in x_labels] for y in y_labels]
	trace1 = {"x": x_labels, "y": y_labels, "z": z_values , "type": "heatmap"}
	
	annotations = [{"x": pred, "y": ref, "font": {"color": "white"}, "showarrow": False, "text": str(matrix[ref][pred]), "xref": "x1",  "yref": "y1"} for pred in x_labels for ref in y_labels]
	data = Data([trace1])
	layout = {"title": "Confusion Matrix", "xaxis": {"title": "Predicted value"}, "yaxis": {"title": "Real value"}, "annotations":annotations}	
	fig = Figure(data=data, layout=layout)
	py.offline.plot(fig, filename=file_path, auto_open=False)	
	if print_latex:
		print('Latex Confusion Matrix:')
		print(' & ' + ' & '.join(x_labels) + '\\\\')
		for y_label in y_labels:
			print(y_label + ' & ' + ' & '.join([str(value) for value in matrix[y_label]]) + '\\\\')
	
	

def viz_docs_rel_difference(refs, preds, out_dir='errors_brat'):
	pred_dict = {pred.id:pred for pred in preds}
	correct_dir, diff_dir = out_dir + '/correct/', out_dir + '/diff/'
	if os.path.exists(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir)			
	os.makedirs(correct_dir)	
	os.makedirs(diff_dir)	
	for ref in refs:
		pred = pred_dict[ref.id]
		correct = deepcopy(ref)
		diff= deepcopy(ref)
		#print(ref.id, len(ref.reverse_span_pair_annotations), len(pred.reverse_span_pair_annotations))
		for pair,labs in pred.reverse_span_pair_annotations.items():
			pred_label = labs[0]
			ref_label = ref.reverse_span_pair_annotations[pair][0]
			#print(pair, labs, ref_label, pred_label)

			if pred_label != ref_label:
				new_label = 'pred:' + pred_label 
				if not new_label in diff.span_pair_annotations:	
					diff.span_pair_annotations[new_label] = []
				diff.span_pair_annotations[new_label].append(pair)
				diff.reverse_span_pair_annotations[pair] = [new_label]
				del correct.reverse_span_pair_annotations[pair]
				if pair in correct.span_pair_annotations[ref_label]:
					correct.span_pair_annotations[ref_label].remove(pair)
			else:
				del diff.reverse_span_pair_annotations[pair]
				if pair in diff.span_pair_annotations[ref_label]:
					diff.span_pair_annotations[ref_label].remove(pair)
		correct.write_to_brat(correct_dir, span_label_filter=None, span_pair_label_filter=None)
		diff.write_to_brat(diff_dir, span_label_filter=None, span_pair_label_filter=None)
				
	
def get_selective_rel_metrics(ref_doc, pred_doc, rels=[], verbose=0, print_pairwise_errors=None):
	
	if print_pairwise_errors:
		if not os.path.exists(print_pairwise_errors):
			os.mkdir(print_pairwise_errors)
		print_pairwise_errors = open(print_pairwise_errors + '/' + ref_doc.id +'.txt', 'w')
	error_string=""	
	rels = set(rels).union(['OTHER'])
	metrics = {rel:{rel:0 for rel in list(rels)+['OTHER']} for rel in rels}
	total, correct = 0.0000000001, 0
	reversed_ref_dict = reverse_dict_list({k:ref_doc.span_pair_annotations[k] for k in ref_doc.span_pair_annotations if k in rels})
	categories = {c:{'mistake':0, 'correct':0} for c in ['EE','TE','DCTE']}
	metrics_per_entity_span = {}
	for pair, anns in reversed_ref_dict.items():
		ref_ann = [ann for ann in anns if ann in rels][0]
		if pair in pred_doc.reverse_span_pair_annotations:
			pred_ann = [ann for ann in pred_doc.reverse_span_pair_annotations[pair] if ann in rels][0]
		else:
			pred_ann = 'OTHER'
			
		category = 'DCTE' if 	pair[0] == (0,0) or pair[1] == (0,0) else ('TE' if pair[0] in ref_doc.span_annotations['EType:TIMEX3'] or pair[1] in ref_doc.span_annotations['EType:TIMEX3'] else 'EE') 
			
		metrics[ref_ann][pred_ann] += 1
		total += 1
		for span in pair:
			if not span in metrics_per_entity_span:
				metrics_per_entity_span[span] = dict([('correct',0), ('mistake',0)])

		if ref_ann == pred_ann:
			correct += 1
			categories[category]['correct'] += 1
			metrics_per_entity_span[pair[0]]['correct']+=1
			metrics_per_entity_span[pair[1]]['correct']+=1
			if not 'cA_'+pred_ann in metrics_per_entity_span[pair[0]]:
				metrics_per_entity_span[pair[0]]['cA_'+pred_ann]=0
			if not pred_ann+ '_cA' in metrics_per_entity_span[pair[1]]:
				metrics_per_entity_span[pair[1]][pred_ann+ '_cA']=0
			metrics_per_entity_span[pair[0]]['cA_'+pred_ann]+=1
			metrics_per_entity_span[pair[1]][pred_ann+ '_cA']+=1
		else:
			categories[category]['mistake'] += 1
			metrics_per_entity_span[pair[0]]['mistake']+=1
			metrics_per_entity_span[pair[1]]['mistake']+=1
			if not 'mA_'+ref_ann in metrics_per_entity_span[pair[0]]:
				metrics_per_entity_span[pair[0]]['mA_'+ref_ann]=0
			if not ref_ann+ '_mA' in metrics_per_entity_span[pair[1]]:
				metrics_per_entity_span[pair[1]][ref_ann+ '_mA']=0
			metrics_per_entity_span[pair[0]]['mA_'+ref_ann]+=1
			metrics_per_entity_span[pair[1]][ref_ann+ '_mA']+=1
			if print_pairwise_errors:
				error_string += '\n\n<X>\t' + str(pair) + '\tref_label:' + ref_ann + '\t' + 'pred_label:' + pred_ann + '\n'
				#print(pair, pair[0])
				error_string += '---<A1>' + ref_doc.span_to_string(pair[0]) + '\t:' + str(pred_doc.reverse_span_annotations[pair[0]])  + '\t'+ (str(ref_doc.span_to_tokens(pair[0],pos=True)) if not pair[0]==(0,0) else 'DCT') +'\n'
				error_string += '---<A2>' +  ref_doc.span_to_string(pair[1]) + '\t:' + str(pred_doc.reverse_span_annotations[pair[1]]) + '\t'+ (str(ref_doc.span_to_tokens(pair[1],pos=True)) if not pair[1]==(0,0) else 'DCT') +'\n'
				
				if not (pair[0] == (0,0) or pair[1] == (0,0)):
					first, second = min([pair[0][0],pair[0][1],  pair[1][0], pair[1][1]]), max([pair[0][0],pair[0][1],  pair[1][0], pair[1][1]])
					error_string += '---<C>' + ref_doc.span_to_string((first, second)) 
				#print(error_string)
				#exit()
	if print_pairwise_errors:
		print_pairwise_errors.write(error_string)		
	
	if verbose:
		print ('---------', ref_doc.id)
		for cat in categories:
			print (cat,':',round(categories[cat]['correct'], 1), '/', round(categories[cat]['correct'] + categories[cat]['mistake'], 1), '=', round(float(categories[cat]['correct']) / (categories[cat]['correct'] + categories[cat]['mistake'] + 0.0000001), 4))
		print ('ACCURACY:',round(correct, 1), '/', round(total,1), '=', round(float(correct) / total, 4))
	return metrics, metrics_per_entity_span
				
		
	
def update_num_dict(d, key, value):
	if not key in d:
		d[key] = value
	else:
		d[key] += 1
	return d
		

def get_evaluation(test_reference, predictions, texts=None, model=None, per_text=False, verbose=True, labels=None):
	print('\nEvaluating:')
	summed_metrics = {}
	mistakes = {}
	if not labels:
		labels = set([label for i in range(len(predictions)) for label in predictions[i]])
	for i in range(len(predictions)):
		
		if texts and model:
			text_metrics, mistakes[i] = get_text_metrics(test_reference[i], predictions[i], text=texts[i], model=model)
		else:
			text_metrics, _ = get_text_metrics(test_reference[i], predictions[i])
			

		for label in labels:
			if not label in summed_metrics:
				summed_metrics[label] = {}
			for k,v in text_metrics[label].items():
				if not k in summed_metrics[label]:
					summed_metrics[label][k] = 0
				summed_metrics[label][k] += len(v) if type(v) == list else v
	metrics = {}
	for label in labels:
		precision = get_precision(summed_metrics[label]['tp'], summed_metrics[label]['fp'])
		recall = get_recall(summed_metrics[label]['tp'], summed_metrics[label]['fn'])
		fmeasure = get_fmeasure(summed_metrics[label]['tp'], summed_metrics[label]['fp'], summed_metrics[label]['fn'])
		if verbose:
			print ('P',str(round(precision,3)).ljust(5), '\tR',str(round(recall,3)).ljust(5), '\tF',str(round(fmeasure,3)).ljust(5), '\t', label)

		metrics[label] = {'tp':summed_metrics[label]['tp'], 'fp':summed_metrics[label]['fp'], 'fn':summed_metrics[label]['fn'], 'precision':precision, 'recall':recall, 'fmeasure':fmeasure}
			

	TP = sum([summed_metrics[label]['tp'] for label in labels])
	FP = sum([summed_metrics[label]['fp'] for label in labels])
	FN = sum([summed_metrics[label]['fn'] for label in labels])
	PRECISION, RECALL, FMEASURE = get_precision(TP, FP), get_recall(TP, FN), get_fmeasure(TP, FP, FN)
	metrics['TOTAL'] = {'precision': PRECISION, 'recall':RECALL, 'fmeasure': FMEASURE,'tp':TP, 'fp':FP, 'fn':FN}
	

			
	if verbose:
		print('---')
		print ('P',str(round(PRECISION,3)).ljust(5), '\tR',str(round(RECALL,3)).ljust(5), '\tF',str(round(FMEASURE,3)).ljust(5), '\t<TOTAL>')
		print ('TP',TP, '\tFP',FP, '\tFN',FN, '\t<TOTAL>')
				
		
	return metrics, mistakes


def get_text_metrics(true_labels, pred_labels, text=None, model=None):
	text_metrics, mistakes = {}, {}
	
	for label in true_labels.keys() + pred_labels.keys():
		if not label in true_labels:
			true_labels[label] = []
		if not label in pred_labels:
			pred_labels[label]= []
			
		label_metrics = get_metrics_from_raw_predictions(true_labels[label],pred_labels[label])
		text_metrics[label] = label_metrics
		
		if text and model:
			mistakes[label] = {}
			mistakes[label]['fp'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['fp']]
			mistakes[label]['fn'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['fn']]
			mistakes[label]['tp'] = [(span_pair, model.preproc_candidate_x(span_pair, text, labeled=True)) for span_pair in label_metrics['tp']]
		
	return text_metrics, mistakes
	
def get_metrics_from_raw_predictions(true, pred):
	tp = get_tp(pred, true)
	fp = get_fp(pred, true)
	fn = get_fn(pred, true)

	precision = get_precision(len(tp),len(fp))
	recall = get_recall(len(tp), len(fn))
	fmeasure = get_fmeasure(len(tp), len(fp), len(fn))	
	return {'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall, 'fmeasure':fmeasure}

def get_tp(pred,true):
	return [span for span in pred if span in true]

def get_fp(pred,true):
	return [span for span in pred if not span in true]

def get_fn(pred, true):
	return [span for span in true if not span in pred]

def get_precision(num_tp, num_fp):
	if num_tp + num_fp ==0:
		return 0.0
	return float(num_tp) / (num_tp + num_fp)
	
def get_recall(num_tp, num_fn):
	if num_tp + num_fn ==0:
		return 0.0
	return float(num_tp) / (num_tp + num_fn)

	
def get_fmeasure(num_tp, num_fp, num_fn, beta=1.0):
	precision = get_precision(num_tp, num_fp)
	recall = get_recall(num_tp, num_fn)
	if precision * recall == 0:
		return 0.0
	else:
		return (1.0+ beta*beta) * ((precision * recall) / (beta*beta 	* precision + recall))
	

def calculate_combined_inv_fmeasure(preds, true, models):
	TP, FP, FN = 0, 0, 0 
	for model in models:
		val_preds = preds[model.name]
		evaluation = calculate_fmeasure(val_preds, true, model)
		TP += evaluation['tp']
		FP += evaluation['fp']
		FN += evaluation['fn']
		print('eval', model.name, evaluation['fmeasure'])
	return 1.0 - get_fmeasure(TP, FP, FN)	


def calculate_fmeasure(preds, true, model):
	evaluation ={'tp':0, 'fp':0, 'fn':0} 
	for i,pred_vec in enumerate(preds):
		
		out_name = model.get_layer_name('output')
		true_vec = true[out_name][i]
		pred_label = model.predict_with_argmax(pred_vec)
		true_label =  model.target_label_vocab_reverse[np.argmax(true_vec)]
		correct = pred_label == true_label
		
		if correct and true_label == 'OTHER': # ignore OTHER labels
			pass
		elif correct and true_label != 'OTHER' :
			evaluation['tp']+=1
		elif not correct and true_label=='OTHER':
			evaluation['fp']+=1
		elif not correct and true_label!='OTHER':
			evaluation['fn']+=1
			if len(model.target_label_vocab) > 2 :
				evaluation['fp']+=1	
	fmeasure = 	get_fmeasure(evaluation['tp'], evaluation['fp'], evaluation['fn'])	
	evaluation.update({'fmeasure':fmeasure})
	return evaluation

def calculate_thresholded_inv_fmeasure(preds, true, model, thresholds=[0.4, 0.5, 0.6]):	
	
	max_f, best_threshold = 0, 0.5
	for threshold_value in thresholds:
		evaluation ={'tp':0, 'fp':0, 'fn':0} 
		for i,pred_vec in enumerate(preds):
		
			out_name = model.get_layer_name('output')
			true_vec = true[out_name][i]
			true_label =  model.target_label_vocab_reverse[np.argmax(true_vec)]

			# prediction with threshold
			pred_label = model.predict_with_threshold(pred_vec, threshold_value)
			
			correct = pred_label == true_label

			if  correct and true_label == 'OTHER': # ignore OTHER labels
				pass
			elif correct and true_label != 'OTHER' :
				evaluation['tp']+=1
			elif not correct and true_label=='OTHER':
				evaluation['fp']+=1
			elif not correct and true_label!='OTHER':
				evaluation['fn']+=1
				if len(model.target_label_vocab) > 2 :
					evaluation['fp']+=1	
						
		fmeasure = get_fmeasure(evaluation['tp'], evaluation['fp'], evaluation['fn'])	
		if fmeasure > max_f:
			max_f = fmeasure
			best_threshold = threshold_value
			
	return 1.0 - max_f, best_threshold

	
def save_entity_error_analysis(docs, entity_error_metrics, output_path):
	#os.makedirs('/'.join(output_path.split('/')[:-1]))
	with open(output_path, 'w') as f:
		for i,doc in enumerate(docs):
			span_metrics = entity_error_metrics[i]
		
			top10_mistakes = sorted([(span,metrics) for (span,metrics) in span_metrics.items()], key=lambda m:m[1]['mistake'], reverse=True)[:10]
			top10_correct = sorted([(span,metrics) for (span,metrics) in span_metrics.items()], key=lambda m:m[1]['correct'], reverse=True)[:10]
		
			f.write('\n\n DOC: ' + str(doc.id) + '\n')
			f.write('\n------- TOP MISTAKES -------\n')
			for span,metrics in top10_mistakes:
				if metrics['mistake']<1:
					continue
				entity_string = doc.text[span[0]:span[1]]
				f.write('\n>>> ' + str(entity_string) + ' <<< \t' + str(metrics['mistake'])+ '\t'+ str(metrics) + '\t' + str(span) +  '\n')
				if span!= (0,0):
					#print(doc.text[span[0]-20:span[1]+20])
					f.write(doc.text[max(span[0]-50,0):min(span[1]+50,len(doc.text))].replace('\n','<newline>').replace(entity_string, '________') + '\n')
				else:
					#print('DCT')
					f.write('<DCT>\n')
			f.write('\n------- TOP CORRECT -------\n')
			for span,metrics in top10_correct:
				if metrics['correct']<1:
					continue
				entity_string = doc.text[span[0]:span[1]]
				f.write('=== ' + str(doc.text[span[0]:span[1]]) + ' === \t' + str(metrics['correct'])+ '\t'+ str(metrics) + '\n')
				if span!= (0,0):
					f.write(doc.text[max(span[0]-50,0):min(span[1]+50,len(doc.text))].replace('\n','<newline>').replace(entity_string, '________') + '\n')
				else:
					f.write('<DCT>\n')

		
