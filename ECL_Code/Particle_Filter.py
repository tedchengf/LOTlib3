from LOTlib3.Samplers.MetropolisHastings import Acontextual_MHS
import numpy as np
import copy

class Particle_Filter():
	# Init
	# 	h0: list of starting hypotheses
	# 	particle_size: int indicating the size of particles
	#	MCMC_step: int indicating the number of steps per rejuvenation
	#	sampler: sampler for MCMC
	def __init__(self, h0, particle_size = 5, MCMC_step = 2, sampler = Acontextual_MHS):
		# Settings
		self.particle_size = particle_size
		self.MCMC_step = MCMC_step
		self.sampler = sampler
		# Current Params
		self.particles = []
		self.weights = []
		self.outputs = None
		self.pred = None
		# History Params
		self.hist_particles = []
		self.hist_weights = []
		# Internal Current Params
		self.__particles = []
		self.__weights = []

		self.__initialize(h0)

	# Apply filtering for one step (one trial)
	# 	ds: list of previous observable data
	#   log: bool of whether to log the particles to history
	def filter_onestep(self, ds, rejuvenation_bound = 0.8, log = False):
		# compute the new weight (as likelihood)
		new_weights = []
		new_outputs = []
		for p in self.__particles:
			new_weights.append(np.exp(p.current_sample.compute_likelihood(ds)))
			new_outputs.append(p.current_sample(*ds[-1].input))
		# update and renormalize weights (this is probably redundent due to resampling)
		new_weights = normalize(np.multiply(new_weights, self.__weights))
		curr_pred = np.dot(new_weights, new_outputs)
		# update parameters
		self.particles = [p.current_sample for p in self.__particles]
		self.weights = new_weights
		self.outputs = new_outputs
		self.pred = curr_pred
		if log == True:
			self.hist_particles.append(self.particles)
			self.hist_weights.append(self.weights)
		# resample and rejuvenate
		self.__particles, self.__weights = self.__resample()
		
		if rejuvenation_bound is not None:
			for p in self.__particles:
				if np.exp(p.current_sample.likelihood) < rejuvenation_bound:
					p.sequential_sample(ds, skip_iter = self.MCMC_step - 1)
		else:
			for p in self.__particles:
				p.sequential_sample(ds, skip_iter = self.MCMC_step - 1)
		return curr_pred

	# A function wrapping filter_onestep to an entire data set
	#	data: list of data
	#	log: bool of whether to log the particles to history
	def filter_block(self, data, memory_limit = 5, rejuvenation_bound = 0.8, log = False):
		preds = []
		for ind in range(len(data)):
			if memory_limit is not None:
				starting_num = ind - memory_limit + 1
				if starting_num < 0: starting_num = 0
				curr_data = data[starting_num:ind+1]
				# print(starting_num, ind+1, len(curr_data))
			else:
				curr_data = data[:ind+1]
			curr_pred = self.filter_onestep(curr_data, rejuvenation_bound=rejuvenation_bound, log = log)
			preds.append(curr_pred)
		return preds

	# Get the prediction for one step without altering the particles
	#	d: a point of observable data
	def pred_onestep(self, d):
		curr_outcomes = [p(*d.input) for p in self.particles]
		return np.dot(self.weights, curr_outcomes)

	# A function wrapping pred_onestep to an entire data set
	#	data: list of data
	def pred_block(self, data):
		return [self.pred_onestep(d) for d in data] 

################################################################################
# Private Methods

	def __initialize(self, h0):
		starting_weights = normalize([np.exp(h.posterior_score) for h in h0])
		starting_hypotheses = np.random.choice(h0, size = self.particle_size, p = starting_weights)
		for ind in range(len(starting_hypotheses)):
			self.__weights.append(np.exp(starting_hypotheses[ind].posterior_score))
			self.__particles.append(self.sampler(starting_hypotheses[ind], None))
			self.particles.append(starting_hypotheses[ind])
		self.__weights = normalize(self.__weights)
		self.weights = copy.copy(self.__weights)

	def __resample(self):
		chosen_particles = np.random.choice(self.__particles, size = self.particle_size, p = self.weights)
		new_particles = self.__copy_particles(chosen_particles)
		return new_particles, [1/self.particle_size]*self.particle_size

	def __copy_particles(self, particles):
		new_particles = []
		for p in particles:
			new_particles.append(self.sampler(copy.copy(p.current_sample), None))
		return new_particles

def normalize(weights):
	try:
		return weights/np.sum(weights)
	except ValueError:
		weights = np.nan_to_num(weights)
		return weights/np.sum(weights)
