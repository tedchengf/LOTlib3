import LOTlib3.Grammar as gmr
import LOTlib3.Primitives.Features
from LOTlib3.Miscellaneous import q
import numpy as np
import pandas as pd
import copy
import os

SHAPE = ["Circle", "Triangle"]
COLOR = ["Red", "Blue"]
SIZE = ["Large", "Small"]


################################################################################
# Define Grammar
################################################################################
obj_grammar = gmr.Grammar()
obj_grammar.add_rule('START', '', ['CONJ'], 1.0)
obj_grammar.add_rule('START', 'True', None, 1.0)
obj_grammar.add_rule('START', 'False', None, 1.0)

obj_grammar.add_rule('CONJ', 'or_', ['CONJ', 'CONJ'], 0.9)
obj_grammar.add_rule('CONJ', '', ['AQUANT'], 1.0)
obj_grammar.add_rule('CONJ', '', ['EQUANT'], 1.0)
obj_grammar.add_rule('CONJ', 'and_', ['AQUANT', 'EQUANT'], 1.0)

# There only need to be one forall clause
obj_grammar.add_rule('AQUANT', 'forall_', ['FUNCTION', 'SET'], 1.0)
# Exists clause can be stacked
obj_grammar.add_rule('EQUANT', '', ['EQUANT_P'], 1.0)
obj_grammar.add_rule('EQUANT', 'and_', ['EQUANT_P', 'EQUANT'], 0.5)
obj_grammar.add_rule('EQUANT_P', 'exists_', ['FUNCTION', 'SET'], 1.0)
obj_grammar.add_rule('EQUANT_P', 'two_type_', ['FUNCTION', 'FUNCTION', 'SET'], 0.2)
obj_grammar.add_rule('SET', 'S', None, 1.0)

obj_grammar.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type = 'OBJECT')
obj_grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 0.5)

obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Red')], 1.0)
obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Blue')], 1.0)
obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Circle')], 1.0)
obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Triangle')], 1.0)
obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Large')], 1.0)
obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Small')], 1.0)

# obj_grammar = gmr.Grammar()
# obj_grammar.add_rule('START', '', ['CONJ'], 1.0)
# obj_grammar.add_rule('START', 'True', None, 1.0)
# obj_grammar.add_rule('START', 'False', None, 1.0)

# obj_grammar.add_rule('CONJ', 'or_', ['CONJ', 'CONJ'], 0.9)
# obj_grammar.add_rule('CONJ', '', ['TQUANT'], 1.0)

# obj_grammar.add_rule("TQUANT", 'two_type_', ['FUNCTION', 'FUNCTION', 'SET'], 1.0)
# obj_grammar.add_rule('SET', 'S', None, 1.0)

# obj_grammar.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type = 'OBJECT')
# obj_grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 0.5)

# obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Red')], 1.0)
# obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Blue')], 1.0)
# obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Circle')], 1.0)
# obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Triangle')], 1.0)
# obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Large')], 1.0)
# obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Small')], 1.0)


################################################################################
# Define Hypothesis
################################################################################
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(BinaryLikelihood, LOTHypothesis):
	def __init__(self, grammar=obj_grammar, **kwargs):
		LOTHypothesis.__init__(self, grammar=grammar, display='lambda S: %s', **kwargs)
		
################################################################################
# Make Objects
################################################################################
import LOTlib3.DataAndObjects as df
from itertools import product, combinations, permutations

objs = df.make_all_objects(shape = ["Circle", "Triangle"], color = ["Red", "Blue"], size = ["Large", "Small"])

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
	
################################################################################
# Read Data
################################################################################
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

sub_data = read_file("/Users/feng/Desktop/ECL_Experiment2/rsps/0220_8/", "Sub_resp.csv", 1)
# burnin_data = sub_data[sub_data["Blc"] == "G1"]
# burnin_seq = burnin_data["Seq"].to_numpy()
# burnin_truth = burnin_data["Truth"].to_numpy()
# burnin_data = make_data()

sub_data = sub_data.iloc[:192]
sequences = sub_data["Seq"].to_numpy()
outcomes = sub_data["Truth"].to_numpy()
data = make_data(sequences, outcomes)

training_data = data[:160]
testing_data = data[160:]

################################################################################
# Running Particle Filter
################################################################################
from LOTlib3 import break_ctrlc
from LOTlib3.Miscellaneous import qq
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler

def normalize(weights):
	return weights/np.sum(weights)

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
	# resample
	new_particles, new_weights = resample(particles, new_weights)
	# apply MCMC
	for p in new_particles:
		p.sequential_sample(curr_data, skip_iter = MCMC_step - 1)
	return new_particles, new_weights, output

def resample(particles, weights):
	new_particles = []
	chosen_particles = np.random.choice(particles, size = len(particles), p = weights)
	for c_p in chosen_particles:
		new_particles.append(MetropolisHastingsSampler(copy.copy(c_p.current_sample), c_p.data))
	return new_particles, [1/len(particles)]*len(particles)

def get_best(particles):
	hs = []
	posteriors = []
	for p in particles:
		hs.append(p.current_sample)
		posteriors.append(np.exp(p.current_sample.posterior_score))
	return hs[np.argmax(posteriors)]

# Establish a pool of particles
h0 = MyHypothesis()
top = TopN(N=50)
thin = 100

sampler = MetropolisHastingsSampler(h0, testing_data)
for i in range(1):
	# h = sampler.sequential_sample(curr_data)
	h = sampler.__next__()
	top << h
	# if i % thin == 0:
	# 	print("#", i, np.exp(h.posterior_score), np.exp(h.prior), np.exp(h.likelihood), qq(h))

pool = []
weights = []
for h in top:
	pool.append(h)
	weights.append(np.exp(h.posterior_score))
	# print(qq(h))
	# print(np.exp(h.posterior_score))
weights = normalize(weights)

print("sampling")
particles = np.random.choice(pool, size = 2, p = weights)
weights = []
for ind in range(len(particles)):
	weights.append(np.exp(particles[ind].posterior_score))
	particles[ind] =  MetropolisHastingsSampler(particles[ind], training_data)

for ind in range(160):
	curr_data = training_data[:ind+1]
	particles, weights, output = filter(particles, weights, curr_data)
	if abs(output - int(outcomes[ind])) > 0.5:
		print(output, int(outcomes[ind]))
		print(curr_data[-1])
	# print(output, int(outcomes[ind]))
print(qq(get_best(particles)))

# for p in particles:
# 	print(type(p))
# 	print(qq(p.current_sample))


# counter = 0
# for h in top:
# 	print(qq(h))
# 	counter += 1
# 	# for ind in range(len(data)):
# 	# 	print(h(*data[ind].input), outcomes[ind])
# print(counter)

exit()