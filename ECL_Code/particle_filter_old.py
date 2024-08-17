import LOTlib3.Grammar as gmr
import LOTlib3.Primitives.Features
from LOTlib3.Miscellaneous import q
import numpy as np
import pandas as pd
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

################################################################################
# Make Objects
################################################################################
import LOTlib3.DataAndObjects as df
from itertools import product, combinations, permutations

def make_data(sequences, outcomes):
	data = []
	for seq, out in zip(sequences, outcomes):
		data.append(df.FunctionData(input=[set([
			make_object(oid) for oid in seq
		])], output=out, alpha=0.99))
	return data

def make_object(obj_id):
	if obj_id == 2:
		return df.Obj(color='Red', shape='Circle', size='Large')
	elif obj_id == 3:
		return df.Obj(color='Red', shape='Circle', size='Small')
	elif obj_id == 5:
		return df.Obj(color='Red', shape='Triangle', size='Large')
	elif obj_id == 7:
		return df.Obj(color='Red', shape='Triangle', size='Small')
	elif obj_id == 11:
		return df.Obj(color='Blue', shape='Circle', size='Large')
	elif obj_id == 13:
		return df.Obj(color='Blue', shape='Circle', size='Small')
	elif obj_id == 17:
		return df.Obj(color='Blue', shape='Triangle', size='Large')
	elif obj_id == 19:
		return df.Obj(color='Blue', shape='Triangle', size='Small')
	else:
		raise RuntimeError("Invalid Object Code:", obj_id)

def read_file(rsp_path, file_name, l_type):	
	with open(rsp_path + file_name, "r") as infile:
		lines = infile.readlines()
		sub_name = lines[0].strip("\n")[14:]
		f_type = lines[1].strip("\n")[14:]
	header = 3
	data = pd.read_csv(rsp_path + file_name, sep = "\t", header = header)
	data["Seq"] = data["Seq"].apply(lambda x: list(map(int, x.split(";"))))
	data.insert(0, "Formula_Type", [f_type]*len(data))
	data.insert(0, "List", [l_type]*len(data))
	data.insert(0, "Subname", [sub_name]*len(data))
	return data

################################################################################
# Define Grammar
################################################################################
def complete_grammar():
	GMR = gmr.Grammar()
	GMR.add_rule('START', '', ['CONJ'], 1.0)
	GMR.add_rule('START', 'True', None, 1.0)
	GMR.add_rule('START', 'False', None, 1.0)

	GMR.add_rule('CONJ', 'or_', ['CONJ', 'CONJ'], 0.9)
	GMR.add_rule('CONJ', '', ['AQUANT'], 1.0)
	GMR.add_rule('CONJ', '', ['EQUANT'], 1.0)
	GMR.add_rule('CONJ', 'and_', ['AQUANT', 'EQUANT'], 1.0)

	# There only need to be one forall clause
	GMR.add_rule('AQUANT', 'forall_', ['FUNCTION', 'SET'], 1.0)
	# Exists clause can be stacked
	GMR.add_rule('EQUANT', '', ['EQUANT_P'], 1.0)
	GMR.add_rule('EQUANT', 'and_', ['EQUANT_P', 'EQUANT'], 0.5)
	GMR.add_rule('EQUANT_P', 'exists_', ['FUNCTION', 'SET'], 1.0)
	GMR.add_rule('EQUANT_P', 'two_type_', ['FUNCTION', 'FUNCTION', 'SET'], 0.1)
	GMR.add_rule('SET', 'S', None, 1.0)

	GMR.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type = 'OBJECT')
	GMR.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 0.5)

	GMR.add_rule('BOOL', 'is_color_', ['OBJECT', q('Red')], 1.0)
	GMR.add_rule('BOOL', 'is_color_', ['OBJECT', q('Blue')], 1.0)
	GMR.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Circle')], 1.0)
	GMR.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Triangle')], 1.0)
	GMR.add_rule('BOOL', 'is_size_', ['OBJECT', q('Large')], 1.0)
	GMR.add_rule('BOOL', 'is_size_', ['OBJECT', q('Small')], 1.0)
	return GMR

def no_type_grammar():
	GMR = gmr.Grammar()
	GMR.add_rule('START', '', ['CONJ'], 1.0)
	GMR.add_rule('START', 'True', None, 1.0)
	GMR.add_rule('START', 'False', None, 1.0)

	GMR.add_rule('CONJ', 'or_', ['CONJ', 'CONJ'], 0.9)
	GMR.add_rule('CONJ', '', ['AQUANT'], 1.0)
	GMR.add_rule('CONJ', '', ['EQUANT'], 1.0)
	GMR.add_rule('CONJ', 'and_', ['AQUANT', 'EQUANT'], 1.0)

	# There only need to be one forall clause
	GMR.add_rule('AQUANT', 'forall_', ['FUNCTION', 'SET'], 1.0)
	# Exists clause can be stacked
	GMR.add_rule('EQUANT', '', ['EQUANT_P'], 1.0)
	GMR.add_rule('EQUANT', 'and_', ['EQUANT_P', 'EQUANT'], 0.5)
	GMR.add_rule('EQUANT_P', 'exists_', ['FUNCTION', 'SET'], 1.0)
	GMR.add_rule('SET', 'S', None, 1.0)

	GMR.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type = 'OBJECT')
	GMR.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 0.5)

	GMR.add_rule('BOOL', 'is_color_', ['OBJECT', q('Red')], 1.0)
	GMR.add_rule('BOOL', 'is_color_', ['OBJECT', q('Blue')], 1.0)
	GMR.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Circle')], 1.0)
	GMR.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Triangle')], 1.0)
	GMR.add_rule('BOOL', 'is_size_', ['OBJECT', q('Large')], 1.0)
	GMR.add_rule('BOOL', 'is_size_', ['OBJECT', q('Small')], 1.0)
	return GMR

# GMR = complete_grammar()
GMR = complete_grammar()

################################################################################
# Define Hypothesis
################################################################################
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib3.Miscellaneous import Infinity, attrmem

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
	def __init__(self, grammar=GMR, **kwargs):
		LOTHypothesis.__init__(self, grammar=grammar, display='lambda S: %s', **kwargs)

	@attrmem('likelihood')
	def compute_likelihood_decay(self, data):
		return

	@attrmem('likelihood')
	def compute_likelihood_local(self, data, shortcut = -Infinity, range = 5):
		return 3.1415926
	
	def compute_posterior(self, data, **kwargs):
		p = self.compute_prior()
		if p > -Infinity:
			l = self.compute_likelihood_local(data, **kwargs)
			return p + l
		else:
			self.likelihood = None
			return -Infinity


################################################################################
# Main
################################################################################

def main():
	# Get Data
	sub_data = read_file("/Users/feng/Desktop/ECL_Experiment2/rsps/0325_1/", "Sub_resp.csv", 1)

	run_subject(sub_data, 100, particle_size = 5, MCMC_steps=2)

	return

################################################################################
# Particle Filter
################################################################################
from LOTlib3 import break_ctrlc
from LOTlib3.Miscellaneous import qq
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler
from multiprocessing import Pool, set_start_method
set_start_method('fork')
from itertools import repeat

def run_subject(sub_data, run_num, particle_size = 5, MCMC_steps = 2):
	sub_data = sub_data.iloc[:192]
	sequences = sub_data["Seq"].to_numpy()
	outcomes = sub_data["Truth"].to_numpy()
	sub_pred = sub_data["Rsp"].to_numpy()
	data = make_data(sequences, outcomes)
	# training data and testing data (G1)
	training_data = data[:160]
	testing_data = data[160:]

	# Establish a pool of particles
	h0 = MyHypothesis()
	top = TopN(N=50)
	# init	
	sampler = MetropolisHastingsSampler(h0, testing_data)
	for i in range(1):
		h = sampler.__next__()
		top << h
	# h_prime = sampler.sequential_sample(testing_data)

	pool = []
	weights = []
	for h in top:
		pool.append(h)
		weights.append(np.exp(h.posterior_score))
	weights = normalize(weights)
	particles = np.random.choice(pool, size = particle_size, p = weights)
	
	global closure
	closure = [particles, training_data, testing_data]
	with Pool() as p:
		results = list(tqdm(p.imap_unordered(simulation, [MCMC_steps]*run_num), total = run_num))
	result_analysis(results, outcomes, sub_pred, testing_data)

	# training_perfs = []
	# training_fits = []
	# testing_perfs = []
	# testing_fits = []
	# rule_distributions = {}
	# for r in results:
	# 	for p in r["particles"]:
	# 		if p in rule_distributions: 
	# 			rule_distributions[p] += 1
	# 		else:
	# 			rule_distributions.update({p: 1})
	# 	training_perfs.append(np.sum(np.square(np.subtract(r["training"], outcomes[:160]))))
	# 	training_fits.append(np.sum(np.square(np.subtract(r["training"], sub_pred[:160]))))
	# 	testing_perfs.append(np.sum(np.square(np.subtract(r["testing"], outcomes[160:]))))
	# 	testing_fits.append(np.sum(np.square(np.subtract(r["testing"], sub_pred[160:]))))
	# for key in rule_distributions: print(key, rule_distributions[key])
	
	# rules = np.array(list(rule_distributions.keys()))
	# freqs = np.array(list(rule_distributions.values()))
	# group_flags = group_hypotheses(rules)
	# for flag in range(1, int(max(group_flags))):
	# 	print(flag)
	# 	curr_rules = rules[group_flags == flag]
	# 	for r in curr_rules: print(r)
	# 	print("===================================================")
	return

	rules, freqs = sort_dict(rule_distributions)
	print("=====================================")
	for ind in range(10):
		print(rules[ind])
		print(freqs[ind])
		print("=====================================")
	best_training_num = np.argmin(training_perfs)

	print(training_fits)
	print(testing_fits)
	print(testing_perfs)

	return

def result_analysis(results, outcomes, sub_pred, testing_data):
	training_perfs = []
	training_fits = []
	testing_perfs = []
	testing_fits = []
	rule_distributions = {}
	for r in results:
		for p in r["particles"]:
			if p in rule_distributions: 
				rule_distributions[p] += 1
			else:
				rule_distributions.update({p: 1})
		training_perfs.append(np.sum(np.square(np.subtract(r["training"], outcomes[:160]))))
		training_fits.append(np.sum(np.square(np.subtract(r["training"], sub_pred[:160]))))
		testing_perfs.append(np.sum(np.square(np.subtract(r["testing"], outcomes[160:]))))
		testing_fits.append(np.sum(np.square(np.subtract(r["testing"], sub_pred[160:]))))
	
	# General Rule Distribution
	print("Particle Distribution")
	rank_rule_dist(rule_distributions, testing_data)

	print("------------------------------------------------------------------------")
	print()
	
	# Best Particle during Training
	print("Best Training Particle")
	best_train = np.argmin(training_fits)
	train_result = results[best_train]["particles"]
	train_distribution = count_rules(train_result, {})
	rank_rule_dist(train_distribution, testing_data)

	print("------------------------------------------------------------------------")
	print()

	# Best Particle during Testing
	print("Best Testing Particle")
	best_test = np.argmin(testing_fits)
	test_result = results[best_test]["particles"]
	test_distribution = count_rules(test_result, {})
	rank_rule_dist(test_distribution, testing_data)

	sequential_fits = []
	sequential_perfs = []
	sequential_ps = []
	for r in results:
		training_fits = np.square(np.subtract(r["training"], sub_pred[:160]))
		testing_fits = np.square(np.subtract(r["testing"], sub_pred[160:]))
		curr_fits = np.concatenate([training_fits, testing_fits])
		curr_fits = segment_timewindows(curr_fits, 40, 5)
		training_perfs = np.square(np.subtract(r["training"], outcomes[:160]))
		testing_perfs = np.square(np.subtract(r["testing"], outcomes[160:]))
		curr_perfs = np.concatenate([training_perfs, testing_perfs])
		curr_perfs = segment_timewindows(curr_perfs, 40, 5)
		sequential_fits.append(curr_fits)
		sequential_perfs.append(curr_perfs)
		sequential_ps.append(segment_timewindows(r["unique_p"], 10, 5))
	
	fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (15,15))
	for sf in sequential_fits:
		ax1.plot(sf, alpha = 0.5)
		ax1.set_title("Particle Fits")
	for sp in sequential_perfs:
		ax2.plot(sp, alpha = 0.5)
		ax2.set_title("Particle Perfs")
	for sps in sequential_ps:
		ax3.plot(sps, alpha = 0.5)
		ax3.set_title("Particle Sizes")
	fig.savefig("Indexs.png", format = "png", dpi = 500, transparent = True)
	

	return

def segment_timewindows(target_arr, width, padding):
	transformed_arr = []
	tw_start = 0
	tw_end = tw_start + width
	while tw_start < len(target_arr):
		if tw_end > len(target_arr):
			transformed_arr.append(np.average(target_arr[tw_start:]))
		else:
			transformed_arr.append(np.average(target_arr[tw_start:tw_end]))
		if tw_end > len(target_arr): break
		tw_start += padding
		tw_end += padding
	return np.array(transformed_arr)

def rank_rule_dist(rules_dict, target_data):
	rules = np.array(list(rules_dict.keys()))
	freqs = np.array(list(rules_dict.values()))
	group_flags = group_hypotheses(rules, target_data)
	final_results = []
	group_distri = []
	for flag in range(1, int(max(group_flags))+1):
		curr_rules = rules[group_flags == flag]
		curr_freqs = freqs[group_flags == flag]
		group_distri.append(np.sum(curr_freqs))
		final_results.append([flag, curr_freqs, curr_rules])
	for ind in np.flip((np.argsort(group_distri))):
		print("------------------------------------------------------------------------")
		print("Group Freq :", np.sum(final_results[ind][1]))
		for rind in range(len(final_results[ind][2])):
			print(final_results[ind][1][rind], ":", final_results[ind][2][rind])

def count_rules(results, r_dist):
	for r in results:
		if r in r_dist:
			r_dist[r] += 1
		else:
			r_dist.update({r: 1})
	return r_dist

def simulation(args):
	starting_hypotheses, training_data, testing_data = closure
	MCMC_steps = args
	unique_p = [len(list(set(starting_hypotheses)))]
	particles = []
	weights = []
	for ind in range(len(starting_hypotheses)):
		weights.append(np.exp(starting_hypotheses[ind].posterior_score))
		particles.append(MetropolisHastingsSampler(starting_hypotheses[ind], training_data))
	weights = normalize(weights)

	training_outputs = []
	for ind in range(len(training_data)):
		curr_data = training_data[:ind+1]
		particles, weights, output, std = filter(particles, weights, curr_data, MCMC_step = MCMC_steps)
		unique_p.append(std)
		training_outputs.append(output)
	testing_outputs = []
	for d in testing_data:
		curr_outcomes = []
		for p in particles:
			curr_outcomes.append(p.current_sample(*d.input))
		testing_outputs.append(np.dot(weights, curr_outcomes))
	result_dict = {
		"training": training_outputs,
		"testing": testing_outputs,
		"unique_p": unique_p,
		"particles": [p.current_sample for p in particles]
	}
	return result_dict

def filter(particles, weights, curr_data, MCMC_step = 2):
	# compute the new weight (as likelihood)
	new_weights = []
	new_outputs = []
	for p in particles:
		new_weights.append(np.exp(p.current_sample.compute_likelihood(curr_data)))
		new_outputs.append(p.current_sample(*curr_data[-1].input))
	# update and renormalize weights (this is probably redundent due to resampling)
	new_weights = np.multiply(new_weights, weights)
	new_weights = normalize(new_weights)
	# obtain expected output
	output = np.dot(new_weights, new_outputs)
	# calculate sample variance
	num_p = len(get_unique(particles))
	# resample
	new_particles, new_weights = resample(particles, new_weights)
	# apply MCMC
	for p in new_particles:
		p.sequential_sample(curr_data, skip_iter = MCMC_step - 1)
	return new_particles, new_weights, output, num_p

def normalize(weights):
	return weights/np.sum(weights)

def resample(particles, weights):
	chosen_particles = np.random.choice(particles, size = len(particles), p = weights)
	# for c_p in chosen_particles:
	#   new_particles.append(MetropolisHastingsSampler(copy.copy(c_p.current_sample),
	#   c_p.data))
	new_particles = copy_particles(chosen_particles)
	return new_particles, [1/len(particles)]*len(particles)

def get_best(particles):
	hs = []
	posteriors = []
	for p in particles:
		hs.append(p.current_sample)
		posteriors.append(np.exp(p.current_sample.posterior_score))
	return hs[np.argmax(posteriors)]

def get_particle_best(particles, weights):
	hs = []
	for p in particles:
		hs.append(p.current_sample)
	return hs[np.argmax(weights)]

def evaluate(particles, weights, data, target):
	predictions = []
	for d in data:
		curr_outcomes = []
		for p in particles:
			curr_outcomes.append(p.current_current_sample(*d.input))
		predictions.append(np.dot(weights, curr_outcomes))
	return np.square(np.subtract(predictions, target))

def get_unique(particles):
	hs = []
	for p in particles:
		hs.append(p.current_sample)
	return list(set(hs))

def copy_particles(particles):
	new_particles = []
	for p in particles:
		new_particles.append(MetropolisHastingsSampler(copy.copy(p.current_sample), p.data))
	return new_particles

def sort_dict(rule_dict):
	keys = np.array(list(rule_dict.keys()))
	vals = np.array(list(rule_dict.values()))
	print(vals)
	order = np.flipud(np.argsort(vals))
	print(order)
	return keys[order], vals[order]

def group_hypotheses(hypotheses, data):
	# if len(hypotheses) == 1:
	# 	return([1])
	h_outcomes = []
	for h in hypotheses:
		curr_outcomes = []
		for d in data:
			curr_outcomes.append(h(*d.input))
		h_outcomes.append(curr_outcomes)

	h_outcomes = np.array(h_outcomes, dtype = int)
	coef_mat = np.corrcoef(h_outcomes)
	# print(coef_mat)
	h_flags = np.zeros(len(hypotheses))
	flag_num = 1
	for s_ind in range(len(h_outcomes)):
		# not assigned a flag
		if h_flags[s_ind] == 0:
			h_flags[s_ind] = flag_num
			for o_ind in range(s_ind+1, len(h_outcomes)):
				# not assigned a flag
				if h_flags[o_ind] == 0:
					if coef_mat[s_ind][o_ind] == 1:
						h_flags[o_ind] = flag_num
			flag_num += 1

	return(h_flags)
	# for ind in range(len(h_outcomes)):
	# 	print(np.dot(h_outcomes[0], h_outcomes[ind])/np.sum(h_outcomes[0]))
	# print(np.matmul(h_outcomes, h_outcomes.T))

if __name__ == "__main__":
	main()