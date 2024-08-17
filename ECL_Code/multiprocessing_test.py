from multiprocessing import Pool
import time
import os
import numpy as np
import itertools
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from tqdm import tqdm

def f(x):
	return x*x

def rand_time(x):
	sleep_time = np.random.randint(0, 10)
	time.sleep(sleep_time)
	return sleep_time

def main():

	timeout_counter = 0
	results = []
	pbar = tqdm(total = 20)
	with ProcessPool() as pool:
		while len(results) < 20:
			future = pool.map(rand_time, itertools.repeat(5, 20 - len(results)), timeout = 5000)
			results_iter = future.result()
			curr_pool_check = True
			while curr_pool_check:
				try:
					result = next(results_iter)
					results.append(result)
					pbar.update(1)
				except StopIteration:
					curr_pool_check = False
				except TimeoutError:
					timeout_counter += 1
	
	print(results)
	print(timeout_counter)

	# with ProcessPool() as pool:
	# 	future = pool.map(rand_time, range(10), timeout = 5)
	# 	results_iter = future.result()
	# 	all_res = []

	# 	while len(all_res) < 10:
	# 		try:
	# 			result = next(results_iter)
	# 			all_res.append(result)
	# 		except TimeoutError:
	# 			print("Time out")
	# 	print(all_res)

		# # make a single worker sleep for 10 seconds
		# processes = [pool.apply_async(rand_time, args = (x,)) for x in range(10)]
		# start_time = time.time()
		# timeout_amount = 5
		# for p in processes:
		# 	try:
		# 		print(p.get(timeout = timeout_amount))
		# 	except TimeoutError:
		# 		print("time out")
		# 	t = time.time() - start_time
		# 	timeout_amount = 5 - t

		# res = pool.apply_async(rand_time, np.arange(10))
		# try:
		# 	print(res.get(timeout=5))
		# except TimeoutError:
		# 	print("We lacked patience and got a multiprocessing.TimeoutError")

	return

if __name__ == "__main__":
	main()