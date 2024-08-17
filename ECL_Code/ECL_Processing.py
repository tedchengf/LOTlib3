################################################################################
# Process Raw Data
################################################################################
import os

REVERSE_LIST = {
				19: 2,
				17: 3,
				13: 5,
				11: 7,
				7: 11,
				5: 13,
				3: 17,
				2: 19,
			   }

def process_data(base_path, file_name, log_name):
	all_data = read_data(base_path, file_name, log_name)
	list1_data = all_data[all_data["List"] == "L1"]
	list2_data = all_data[all_data["List"] == "L2"]
	list2_data.loc[:, "Seq"] = list2_data["Seq"].apply(reverse_stim)
	all_data = pd.concat([list1_data, list2_data], axis = 0)
	# all_data.to_csv("all_data.csv")
	return all_data

def read_data(base_path, file_name, log_name):
	# Get subject logs
	sublog = {}
	subcheck = {}
	# with open(os.path.join(script_dir, "input/Sub_log.txt"), "r") as infile:
	with open(log_name, "r") as infile:
		lines = infile.readlines()
		for l in lines:
			curr_l = l.strip("\n")
			curr_l = curr_l.split("\t")
			sublog.update({curr_l[0]:curr_l[1:]})
			subcheck.update({curr_l[0]:curr_l[1]})

	# Get the data
	all_data = []
	for sub_dir in os.listdir(base_path):
		if sub_dir != ".DS_Store":
			sub_list = sublog[sub_dir][1]
			sub_data = read_file(base_path+sub_dir+"/", file_name, sub_list)
			if sub_data["Subname"].unique()[0] != sub_dir:
				raise RuntimeError("Mismatch between sublog and subject data at", sub_dir, "with subname", sub_data["Subname"].unique()[0])
			all_data.append(sub_data)
	all_data = pd.concat(all_data, axis = 0)

	return all_data

def reverse_stim(seq):
	for ind in range(len(seq)):
		seq[ind] = REVERSE_LIST[seq[ind]]
	return seq

################################################################################
# Make Objects
################################################################################
import LOTlib3.DataAndObjects as df
import pandas as pd

class ECL_FunctionData(df.FunctionData):
	def __init__(self, id, input, output, **kwargs):
		"""Creates a new FunctionData object; input must be either a list or a tuple."""
		# Since we apply to this, it must be a list
		assert isinstance(input, list) or isinstance(input, tuple), "FunctionData.input must be a list!"
		self.id = id
		self.input = input
		self.output = output
		self.__dict__.update(kwargs)

def make_data(sequences, outcomes):
	data = []
	for seq, out in zip(sequences, outcomes):
		data.append(ECL_FunctionData(tuple(seq), input=[set([
			make_object(oid) for oid in seq
		])], output=out, alpha=0.99))
		# data.append(df.FunctionData(input=[make_object(oid) for oid in seq], output = out, alpha=0.99))
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