import LOTlib3.Grammar as gmr
import LOTlib3.Primitives.Features
from LOTlib3.Miscellaneous import q
import numpy as np
import pandas as pd
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
obj_grammar.add_rule('SET', 'S', None, 1.0)

# obj_grammar.add_rule('CONJ', 'and_', ['QUANT', 'CONJ'], 0.2)
# obj_grammar.add_rule('CONJ', '', ['QUANT'], 1.0)

# obj_grammar.add_rule('QUANT', 'exists_', ['FUNCTION', 'SET'], 1.0)
# obj_grammar.add_rule('QUANT', 'forall_', ['FUNCTION', 'SET'], 1.0)
# obj_grammar.add_rule('SET', 'S', None, 1.0)

obj_grammar.add_rule('FUNCTION', 'lambda', ['BOOL'], 1.0, bv_type = 'OBJECT')
obj_grammar.add_rule('BOOL', 'and_', ['BOOL', 'BOOL'], 0.5)

obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Red')], 1.0)
obj_grammar.add_rule('BOOL', 'is_color_', ['OBJECT', q('Blue')], 1.0)
obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Circle')], 1.0)
obj_grammar.add_rule('BOOL', 'is_shape_', ['OBJECT', q('Triangle')], 1.0)
obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Large')], 1.0)
obj_grammar.add_rule('BOOL', 'is_size_', ['OBJECT', q('Small')], 1.0)

# rules = set([])
# for _ in range(100):
#     curr_rule = obj_grammar.generate()
#     if curr_rule not in rules: rules.add(curr_rule)

# rules = list(rules)
# for r in rules: 
#     print(r)
# exit()

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

sub_data = read_file("/Users/feng/Desktop/ECL_Experiment2/rsps/0220_1/", "Sub_resp.csv", 1)
sub_data = sub_data[sub_data["Blc"] == '1']
sequences = sub_data["Seq"].to_numpy()
outcomes = sub_data["Truth"].to_numpy()

data = make_data(sequences, outcomes)
# for d in data: print(d)

from LOTlib3 import break_ctrlc
from LOTlib3.Miscellaneous import qq
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler


h0 = MyHypothesis()
top = TopN(N=10)
thin = 10000

for i, h in enumerate(break_ctrlc(MetropolisHastingsSampler(h0, data))):
	top << h
	if i % thin == 0:
		print("#", i, np.exp(h.posterior_score), np.exp(h.prior), np.exp(h.likelihood), qq(h))
	

for h in top:
	print(qq(h))

	for ind in range(len(data)):
		print(h(*data[ind].input), outcomes[ind])

	exit()
# for h in top:
# 	print(h.posterior_score, h.prior, h.likelihood, qq(h))