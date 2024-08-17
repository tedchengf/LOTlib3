import LOTlib3.Grammar as gmr
import LOTlib3.DataAndObjects as df
import LOTlib3.Primitives.Features
from LOTlib3.DefaultGrammars import DNF
from LOTlib3.Miscellaneous import q
import numpy as np

SHAPE = ["Circle", "Triangle"]
COLOR = ["Red", "Blue"]
SIZE = ["Large", "Small"]

# def make_data(n=1, alpha=0.999):
#     return [
#         		df.FunctionData(input=[df.Obj(shape='square', color='red')], output=True, alpha=alpha),
#        			df.FunctionData(input=[df.Obj(shape='square', color='blue')], output=True, alpha=alpha),
#         		df.FunctionData(input=[df.Obj(shape='triangle', color='blue')], output=True, alpha=alpha),
#         		df.FunctionData(input=[df.Obj(shape='triangle', color='red')], output=True, alpha=alpha)
#             ]*n

objs = df.make_all_objects(shape = ["Circle", "Triangle"], color = ["Red", "Blue"], size = ["Large", "Small"])
data = []
for obj in objs:
    data.append(df.FunctionData(input = [obj], output = True))

grammar = DNF 

grammar.add_rule('PREDICATE', 'is_color_', ['x', 'COLOR'], 1.0)
grammar.add_rule('PREDICATE', 'is_shape_', ['x', 'SHAPE'], 1.0)
grammar.add_rule('PREDICATE', 'is_size_', ['x', 'SIZE'], 1.0)
grammar.add_rule('COLOR', q('Red'), None, 1.0)
grammar.add_rule('COLOR', q('Blue'), None, 1.0)
grammar.add_rule('SHAPE', q('Circle'), None, 1.0)
grammar.add_rule('SHAPE', q('Triangle'), None, 1.0)
grammar.add_rule('SIZE', q('Large'), None, 1.0)
grammar.add_rule('SIZE', q('Small'), None, 1.0)

from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Priors.RationalRules import RationaRulesPrior
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood

class MyHypothesis(RationaRulesPrior, BinaryLikelihood, LOTHypothesis):
    def __init__(self, **kwargs):
        LOTHypothesis.__init__(self, grammar=grammar, **kwargs)
        self.rrAlpha=2.0

# curr_h = MyHypothesis(value = "is_color_(x, 'Red')")
# print(curr_h.value)
# print(curr_h(objs[0]))

# for _ in range(10):
#     curr_h = MyHypothesis()
#     print(curr_h.value)
#     print(curr_h(objs[0]))

for nt in grammar.nonterminals():
    print(nt)
print("-------")

r = grammar.get_rules("START")
for rule in r: print(rule)
fn = r[1].make_FunctionNodeStub(grammar, None)
print(fn)
print(fn.args)
exit()

rules = set([])
for _ in range(30):
    curr_rule = grammar.generate()
    if curr_rule not in rules: rules.add(curr_rule)

rules = list(rules)
for r in rules: 
    print(r)
    print(type(r))
    # print(grammar.pack_ascii(r))
# print(rules[0])
# print(grammar.pack_ascii(rules[0]))

# # print(objs[0])

# for r in rules: 
#     print(r)
#     print(r(objs[0]))
      