import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import pickle
import os
import math
import time

import ECL_Processing
import ECL_Grammar
import Particle_Filter

CURR_DIR = os.path.dirname(__file__)
HASH_DICT_PATH = os.path.join(CURR_DIR, "input/hash_dict")
SUB_DATA_DIR = os.path.join(CURR_DIR, "input/rsps/")
SUB_LOG_PATH = os.path.join(CURR_DIR, "input/Sub_log.txt")
OUTPUT_PATH = os.path.join(CURR_DIR, "output/")

################################################################################
# Settings and Main
################################################################################
# Number of simulations per subject
RUN_NUM = 800
# Size of the particle in each simulation
PARTICLE_SIZE = 5
# Number of MCMC steps to run for each round of rejuvenation
MCMC_STEPS = 4
# Maximum number of trials each filter can access (hard limit)
MEMORY_LIMIT = None
# The rate of decay for previous trials. The discounting function is 
# 	e^(-DECARY_RATE)
DECAY_RATE = 0.15
# The number of trial to skip before the discounting function is applied
DECAY_BUFFER = 0
# The likelihood correctness of each particle beneath which rejunivation will
# 	be applied 
REJUVENATION_BOUND = 0.8
# The number of steps of metropolis hasting used to generate the starting
# 	hypothesis pool
STARTING_SIZE = 100
# The timeout limit for each filter when using multiprocessing (in seconds)
TIMEOUT = 30

def main():
	
	all_data = ECL_Processing.process_data(SUB_DATA_DIR, "Sub_resp.csv", SUB_LOG_PATH)

	# test_rules = generate_rules(all_data, steps = 5000000)
	# with open("./test_rules", "wb") as outfile:
	# 	pickle.dump(test_rules, outfile)

	# with open("./test_rules", "rb") as infile:
	# 	rules = pickle.load(infile)
	# for block_cond in ["T2", "B1", "B2", "S1"]:
	# 	fit_sub_finalrules(all_data, rules, block_cond)

	# for cond in ["B1", "B2", "S1"]:
	# 	cond_results = reanalyze_results(all_data, cond)
	# 	print("Fits:", np.average(cond_results["Fits"]))
	# 	print("Perfs:", np.average(cond_results["Perf"]))

	with open(HASH_DICT_PATH, "rb") as infile:
		hash_dict = pickle.load(infile)
	
	# cond_results = reanalyze_results(all_data, "T1", res_dir = "./test results/", hash_dict=hash_dict)
	# print("Fits:", np.average(cond_results["Fits"]))
	# print("Perfs:", np.average(cond_results["Perf"]))

	# for k in hash_dict:
	# 	print(k, hash_dict[k])

	hash_dict = {}

	for block_cond in ["T2"]:
		curr_data = all_data[all_data["Formula_Type"] == block_cond]

		sub_results = []
		pbar = tqdm(total=len(curr_data["Subname"].unique()), position=0, leave=True, desc="Sub Iter")
		for ind, subname in enumerate(curr_data["Subname"].unique()):
			# if subname != "1116_4": continue
			# print(subname)
			sub_data = curr_data[curr_data["Subname"] == subname]
			# curr_result = run_subject_nonpar(sub_data, hash_dict=hash_dict, run_num = RUN_NUM, particle_size = PARTICLE_SIZE,MCMC_steps = MCMC_STEPS, memory_limit = MEMORY_LIMIT, starting_size = STARTING_SIZE, log = True)
			# with open("./run_0806/" + subname, "wb") as outfile:
			# 	pickle.dump(curr_result, outfile)
			sub_results.append(run_subject(sub_data, hash_dict=hash_dict, run_num = RUN_NUM, particle_size = PARTICLE_SIZE,MCMC_steps = MCMC_STEPS, memory_limit = MEMORY_LIMIT, starting_size = STARTING_SIZE, log = True))
			with open(OUTPUT_PATH + subname, "wb") as outfile:
				pickle.dump(sub_results[-1][0], outfile)
			pbar.update(1)
			
			# if ind > 5: break

	with open(OUTPUT_PATH + "finish_flag.txt", 'w') as outfile:
		outfile.write("finished")

		# model_fits = []
		# model_accs = []
		# for sub_res in sub_results:
		# 	run_results, outcomes, subpreds = sub_res
		# 	for run_res in run_results:
		# 		model_fits.append(np.sum(np.square(np.subtract(run_res["testing"], subpreds[160:]))))
		# 		model_accs.append(np.sum(np.square(np.subtract(run_res["testing"], outcomes[160:]))))
		# print(block_cond)
		# print("Model Fits:", np.average(model_fits))
		# print("Model Perfs:", np.average(model_accs))

	with open(HASH_DICT_PATH, "wb") as outfile:
		pickle.dump(hash_dict, outfile)

	# sub_data = ECL_Processing.read_file("/Users/feng/Desktop/ECL_Experiment2/rsps/0229_1/", "Sub_resp.csv", 1)
	# sub_results, outcomes, sub_pred = run_subject(sub_data, run_num = RUN_NUM, particle_size = PARTICLE_SIZE,MCMC_steps = MCMC_STEPS, memory_limit = MEMORY_LIMIT, starting_size = STARTING_SIZE, log = True)
	# with open("./0221_3", "wb") as outfile:
	# 	pickle.dump(sub_results, outfile)

	# with open("./0229_1", "rb") as infile:
	# 	sub_results = pickle.load(infile)
	# # Segment Subject Data
	# sub_data = sub_data.iloc[:192]
	# sequences = sub_data["Seq"].to_numpy()
	# outcomes = sub_data["Truth"].to_numpy()
	# sub_pred = sub_data["Rsp"].to_numpy()
	# data = ECL_Processing.make_data(sequences, outcomes) 
	# testing_data = data[160:]
	# result_analysis(sub_results, outcomes, sub_pred, testing_data)

	# run_subject_nonpar(sub_data, run_num = 1, particle_size = PARTICLE_SIZE, MCMC_steps = MCMC_STEPS, memory_limit = MEMORY_LIMIT, starting_size = STARTING_SIZE, rejuvenation_bound = REJUVENATION_BOUND)
	return

################################################################################
# Define Hypothesis and Likelihood
################################################################################
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
from LOTlib3.Miscellaneous import Infinity, attrmem
from math import log
from LOTlib3.Eval import RecursionDepthException
from LOTlib3.Miscellaneous import Infinity

GMR = ECL_Grammar.complete_grammar()

class BinaryLikelihood_Discount(object):
	def compute_single_likelihood(self, datum, weight):
		try: 
			# outcome * weight + (1 - weight)
			# the smaller the weight, the closer the result to 1 regardless of
			# outcome
			return log((datum.alpha * (self(*datum.input) == datum.output) + (1.0-datum.alpha) / 2.0) * weight + (1 - weight))
		except RecursionDepthException as e:
			return -Infinity

class ECL_Hypothesis(BinaryLikelihood_Discount, LOTHypothesis):
	def __init__(self, grammar = GMR, **kwargs):
		LOTHypothesis.__init__(self, grammar=grammar, display='lambda S: %s', **kwargs)

	@attrmem('likelihood')
	def compute_likelihood(self, data, shortcut=-Infinity, decay_rate = DECAY_RATE, decay_buffer = DECAY_BUFFER):
		pos_index = np.concatenate((np.flipud(np.arange(len(data) - decay_buffer)), [0]*decay_buffer))
		weights = np.exp(-pos_index)
		ll = 0.0
		for ind, datum in enumerate(data):
			ll += self.compute_single_likelihood(datum, weights[ind])
		return ll
	

################################################################################
# Parallel Filtering
################################################################################
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler, Acontextual_MHS
from multiprocessing import Pool, set_start_method
set_start_method('fork')

def structure_subdata(sub_data):
	sub_data = sub_data.iloc[:192]
	sequences = sub_data["Seq"].to_numpy()
	outcomes = sub_data["Truth"].to_numpy()
	sub_pred = sub_data["Rsp"].to_numpy()
	data = ECL_Processing.make_data(sequences, outcomes) 
	training_data = data[:160]
	testing_data = data[160:]

	return outcomes, sub_pred, training_data, testing_data

# Main function for running a single subject in parallel
def run_subject(sub_data, hash_dict = None, run_num = 100, particle_size = 5, MCMC_steps = 2, memory_limit = 5, starting_size = 100, rejuvenation_bound = 0.8, log = False):

	outcomes, sub_pred, training_data, testing_data = structure_subdata(sub_data)

	# Init Hypotheses
	h_start = ECL_Hypothesis(GMR)
	top = TopN(N=starting_size)
	sampler = MetropolisHastingsSampler(h_start, [training_data[10]])
	for i in range(starting_size):
		h = sampler.__next__()
		top << h
	h0 = [h for h in top]

	# Parallal Application with Timeout
	global CLOSURE
	CLOSURE = [h0, training_data, testing_data]
	pbar = tqdm(total = run_num, position=1, leave=False, desc="Run iter")
	results = []
	if log == False:
		target_func = single_filter
	else:
		target_func = single_filter_logged
	with ProcessPool() as pool:
		while len(results) < run_num:
			future = pool.map(target_func, itertools.repeat((particle_size, MCMC_steps, memory_limit, rejuvenation_bound), run_num - len(results)), timeout = TIMEOUT)
			results_iter = future.result()
			curr_pool_check = True
			while curr_pool_check:
				try:
					run_res = next(results_iter)
					results.append(run_res)
					pbar.update(1)
				except StopIteration:
					curr_pool_check = False
				except TimeoutError:
					print("timeout")

	# # Parallal Application
	# global CLOSURE
	# CLOSURE = [h0, training_data, testing_data]
	# with Pool() as pool:
	# 	if log == False:
	# 		results = list(tqdm(pool.imap_unordered(single_filter, [(particle_size, MCMC_steps, memory_limit, rejuvenation_bound)]*run_num), total = run_num, position=1, leave=False, desc="Run iter"))
	# 	else:
	# 		results = list(tqdm(pool.imap_unordered(single_filter_logged, [(particle_size, MCMC_steps, memory_limit, rejuvenation_bound)]*run_num), total = run_num, position=1, leave=False, desc="Run iter"))
	# 		# print(results[0]["particles"])
	# 		# print("")
	# 		# print(np.array(results[0]["hist_particles"]))
	# 		# print(np.array(results[0]["hist_particles"]).shape)

	for ind in range(len(results)):
		curr_result = results[ind]
		curr_result["particles"] = condense_particles(curr_result["particles"], hash_dict)
		curr_hist = curr_result["hist_particles"]
		new_hist = []
		for trial in curr_hist: new_hist.append(condense_particles(trial, hash_dict))
		curr_result["hist_particles"] = new_hist

	# result_analysis(results, outcomes, sub_pred, testing_data)
	return results, outcomes, sub_pred

# Function for running a single subject. Mainly for testing
def run_subject_nonpar(sub_data, hash_dict = None, run_num = 100, particle_size = 5, MCMC_steps = 2, memory_limit = 5, starting_size = 100, rejuvenation_bound = 0.8, log = False):
	outcomes, sub_pred, training_data, testing_data = structure_subdata(sub_data)

	# Init Hypotheses
	h_start = ECL_Hypothesis(GMR)
	top = TopN(N=starting_size)
	sampler = MetropolisHastingsSampler(h_start, [training_data[1]])
	for i in range(starting_size):
		h = sampler.__next__()
		top << h
	h0 = [h for h in top]

	# Parallal Application
	global CLOSURE
	CLOSURE = [h0, training_data, testing_data]
	results = []
	test_time = open("test_times.txt", "a+")
	for ind in tqdm(range(run_num), position=1, leave=False, desc="Run iter"):
		start = time.time()
		if log == False:
			results.append(single_filter((particle_size, MCMC_steps, memory_limit, rejuvenation_bound)))
		else:
			results.append(single_filter_logged((particle_size, MCMC_steps, memory_limit, rejuvenation_bound)))
		end = time.time()
		test_time.write("Round " + str(ind) + ": " + str(end - start) + "\n")
	test_time.close()

	for ind in range(len(results)):
		curr_result = results[ind]
		curr_result["particles"] = condense_particles(curr_result["particles"], hash_dict)
		curr_hist = curr_result["hist_particles"]
		new_hist = []
		for trial in curr_hist: new_hist.append(condense_particles(trial, hash_dict))
		curr_result["hist_particles"] = new_hist

	return results, outcomes, sub_pred

# Core function for each filter
def single_filter(args):
	particle_size, MCMC_steps, memory_limit, rejuvenation_bound = args
	h0, training_data, testing_data = CLOSURE
	curr_filter = Particle_Filter.Particle_Filter(h0, particle_size=particle_size, MCMC_step=MCMC_steps, sampler=Acontextual_MHS)
	training_preds = curr_filter.filter_block(training_data, memory_limit=memory_limit, rejuvenation_bound=rejuvenation_bound)
	final_particles = curr_filter.particles
	testing_preds = curr_filter.pred_block(testing_data)
	return {
			"training": training_preds, 
	 		"testing": testing_preds, 
			"particles": final_particles
		   }

# Core function for each filter, logged version
def single_filter_logged(args):
	particle_size, MCMC_steps, memory_limit, rejuvenation_bound = args
	h0, training_data, testing_data = CLOSURE
	curr_filter = Particle_Filter.Particle_Filter(h0, particle_size=particle_size, MCMC_step=MCMC_steps, sampler=Acontextual_MHS)
	training_preds = curr_filter.filter_block(training_data, memory_limit=memory_limit, rejuvenation_bound=rejuvenation_bound, log = True)
	final_particles = curr_filter.particles
	hist_particles = curr_filter.hist_particles
	hist_weights = curr_filter.hist_weights
	testing_preds = curr_filter.pred_block(testing_data)
	return {
			"training": training_preds, 
	 		"testing": testing_preds, 
			"particles": final_particles,
			"hist_particles": hist_particles,
			"hist_weights": hist_weights
		   }

################################################################################
# Analyze Results
################################################################################
def result_analysis(results, outcomes, sub_pred, testing_data):
	training_perfs = []
	training_fits = []
	testing_perfs = []
	testing_fits = []
	alt_training_fits = []
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
		alt_training_fits.append(np.sum(np.square(np.subtract(r["training"][80:], sub_pred[80:160]))))
	
	# General Rule Distribution
	print("Particle Distribution")
	freqs, rule_freqs, rules = simplify_rules(rule_distributions, testing_data)
	print(freqs)
	print(rule_freqs.flatten())
	print(rules.flatten())


	rank_rule_dist(rule_distributions, testing_data)

	print("------------------------------------------------------------------------")
	print()
	
	# Best Particle during Training
	print("Best Training Particle")
	# best_train = np.argmin(alt_training_fits)
	# train_result = results[best_train]["particles"]
	# train_distribution = count_rules(train_result, {})
	# rank_rule_dist(train_distribution, testing_data)
	best_trains = np.argsort(alt_training_fits)[:3]
	train_distribution = {}
	for ind in best_trains:
		train_distribution = count_rules(results[ind]["particles"], train_distribution)
	# train_result = results[best_train]["particles"]
	# train_distribution = count_rules(train_result, {})
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
	
	fig, (ax1, ax2) = plt.subplots(2,1, figsize = (15,15))
	for sf in sequential_fits:
		ax1.plot(sf, alpha = 0.5)
		ax1.set_title("Particle Fits")
	for sp in sequential_perfs:
		ax2.plot(sp, alpha = 0.5)
		ax2.set_title("Particle Perfs")
	fig.savefig("Indexs.png", format = "png", dpi = 500, transparent = True)

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
					if math.isclose(coef_mat[s_ind][o_ind], 1):
						h_flags[o_ind] = flag_num
			flag_num += 1
	return(h_flags)

def simplify_rules(rules_dict, target_data, group_length = 1):
	rules = np.array(list(rules_dict.keys()))
	freqs = np.array(list(rules_dict.values()))
	group_flags = group_hypotheses(rules, target_data)
	all_freqs = []
	group_rules = []
	group_freqs = []
	for flag in range(1, int(max(group_flags))+1):
		curr_rules = rules[group_flags == flag]
		curr_freqs = freqs[group_flags == flag]
		group_order = np.flip(np.argsort(curr_freqs))
		all_freqs.append(np.sum(curr_freqs))
		max_ind = min(group_length, len(curr_freqs))
		group_freqs.append(curr_freqs[group_order][:max_ind])
		group_rules.append(curr_rules[group_order][:max_ind])

	all_order = np.flip((np.argsort(all_freqs)))
	return np.take(all_freqs, all_order, axis = 0), np.take(group_freqs, all_order, axis = 0), np.take(group_rules, all_order, axis = 0)


def rank_rule_dist(rules_dict, target_data, outfile):
	rules = np.array(list(rules_dict.keys()))
	freqs = np.array(list(rules_dict.values()))
	group_flags = group_hypotheses(rules, target_data)
	final_results = []
	group_distri = []
	outfile.write("Unique Groups:" + str(int(max(group_flags))) + "\n")
	outfile.write("All Frequency:" + str(int(sum(freqs))) + "\n")
	for flag in range(1, int(max(group_flags))+1):
		curr_rules = rules[group_flags == flag]
		curr_freqs = freqs[group_flags == flag]
		group_distri.append(np.sum(curr_freqs))
		final_results.append([flag, curr_freqs, curr_rules])
	for ind in np.flip((np.argsort(group_distri))):
		outfile.write("--------------------------------------------------------------------------------")
		outfile.write("\n")
		outfile.write("Group Freq : " + str(np.sum(final_results[ind][1])) + "\n")
		for rind in range(len(final_results[ind][2])):
			outfile.write(str(final_results[ind][1][rind]) + " : " + str(final_results[ind][2][rind]) + "\n")
			# print(final_results[ind][1][rind], ":", final_results[ind][2][rind])

def count_rules(results, r_dist):
	for r in results:
		if r in r_dist:
			r_dist[r] += 1
		else:
			r_dist.update({r: 1})
	return r_dist

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

def condense_particles(particles, hash_dict):
	c_particles = []
	for p in particles:
		p_hash = p.__hash__()
		c_particles.append(p_hash)
		if p_hash not in hash_dict:
			hash_dict.update({p_hash: p})
	return c_particles

def reconstruct_particles(particles, hash_dict):
	r_particles = []
	for p in particles:
		r_particles.append(hash_dict[p])
	return r_particles


def reanalyze_results(all_data, target_condition, res_dir = "./Filter Results/", hash_dict = None):
	curr_data = all_data[all_data["Formula_Type"] == target_condition]
	sub_names = set(curr_data["Subname"].unique())

	cond_results = {
		"Fits": [],
		"Perf": [],
		"Rule": {},
	}
	pbar1 = tqdm(total=len(curr_data["Subname"].unique()), position=0, leave=True, desc="Sub Iter")
	for sname in os.listdir(res_dir):
		if sname in sub_names:
			sub_data = curr_data[curr_data["Subname"] == sname]
			outcomes, sub_pred, training_data, testing_data = structure_subdata(sub_data)
			with open(res_dir + sname, "rb") as infile:
				sub_file = pickle.load(infile)
				for run_res in sub_file:
					testing = run_res["testing"]
					rules = run_res["particles"]
					if hash_dict is not None:
						rules = reconstruct_particles(rules, hash_dict)
					for r in rules:
						if r in cond_results["Rule"]:
							cond_results["Rule"][r] += 1
						else:
							cond_results["Rule"].update({r: 1})
					cond_results["Fits"].append(np.sum(np.square(np.subtract(testing, sub_pred[160:]))))
					cond_results["Perf"].append(np.sum(np.square(np.subtract(testing, outcomes[160:]))))
			pbar1.update(1)

	with open("./"+target_condition+".txt", "w") as outfile:
		outfile.write("Fits:" + str(np.average(cond_results["Fits"])) + "\n")
		outfile.write("Perf:" + str(np.average(cond_results["Perf"])) + "\n")
		rank_rule_dist(cond_results["Rule"], testing_data, outfile)
	return cond_results

def generate_rules(data, steps = 100000):
	sub_name = data["Subname"].unique()[0]
	sub_data = data[data["Subname"] == sub_name]
	outcomes, sub_pred, training_data, testing_data = structure_subdata(sub_data)

	all_rules = {}
	h_start = ECL_Hypothesis(GMR)
	sampler = MetropolisHastingsSampler(h_start, testing_data)
	pbar = tqdm(total=steps, position=0, leave=True, desc="Data Iter")
	for _ in range(steps):
		h = sampler.__next__()
		if h not in all_rules:
			all_rules.update({h:1})
		pbar.update(1)
	
	all_freqs, group_freqs, rules = simplify_rules(all_rules, testing_data)
	rules = rules.flatten()

	test = rules[50]
	# for d in testing_data:
	# 	print(test(*d.input))

	return rules

def fit_sub_finalrules(data, rules, target_condition):
	data = data[data["Formula_Type"] == target_condition]
	final_rules = {}
	pbar = tqdm(total=len(data["Subname"].unique()), position=0, leave=True, desc="Sub Iter")
	for sub_name in data["Subname"].unique():
		sub_data = data[data["Subname"] == sub_name]
		outcomes, sub_pred, training_data, testing_data = structure_subdata(sub_data)
		subtest_fit = []
		for r in rules:
			r_res = []
			for d in testing_data: r_res.append(r(*d.input))
			subtest_fit.append(percentage_fit(sub_pred[160:], r_res))
		best_ind = np.argmax(subtest_fit)
		best_rule = rules[best_ind]
		if rules[best_ind] not in final_rules:
			final_rules.update({rules[best_ind]: 1})
		else:
			final_rules[rules[best_ind]] += 1
		pbar.update(1)
	
	list_rules = np.array(list(final_rules.keys()))
	list_freqs = np.array(list(final_rules.values()))
	order = np.flip(np.argsort(list_freqs))
	with open(target_condition + "_optimal.txt", "w") as outfile:
		outfile.write("Unique Groups:" + str(len(rules)) + "\n")
		outfile.write("All Frequency:" + str(int(sum(list_freqs))) + "\n")
		for ind in order:
			outfile.write("--------------------------------------------------------------------------------")
			outfile.write("\n")
			outfile.write("Group Freq : " + str(list_freqs[ind]) + "\n")		
			outfile.write(str(list_rules[ind]) + "\n")
	return final_rules

def percentage_fit(list1, list2):
	list1 = np.array(list1, dtype=int)
	list2 = np.array(list2, dtype=int)
	return 1 - (np.sum(np.abs(list1 - list2))/len(list1))

if __name__ == "__main__":
	main()