################################################################################
# Define Primitives
################################################################################
import itertools
from LOTlib3.Eval import primitive
from LOTlib3.Miscellaneous import q

@primitive
def two_type_(F1, F2, S):
	for seq_assign in itertools.permutations(S, 2):
		if F1(seq_assign[0]) and F2(seq_assign[1]): return True
	return False

################################################################################
# Define Grammar
################################################################################
import LOTlib3.Grammar as gmr

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
