
from itertools import chain
import numpy as np

class Partitioner:

	inf = float('inf')
	
	def __init__(self, data, block_size = 8):

		self.block_size = block_size

		self.data = data
		self.size = len(data)

	def set_prefix_sum(self, data):
		""" O(N) Get prefix sum array of immediate changes """

		self.prefix_sum = [0]

		for i in range(1, self.size):

			diff = abs(self.data[i] - self.data[i - 1])
			self.prefix_sum.append(self.prefix_sum[i - 1] + diff)
		
		return self.prefix_sum

	def get_diff(self, l, r):
		""" O(1) Get changes given subsection inclusive """
		return self.prefix_sum[r] - self.prefix_sum[l]


	def cluster_percentage(self):

		self.diffs = [0.0]

		size = len(self.clusters)

		for i in range(1, size):

			diff = abs(self.data[i][0] - self.data[i - 1][-1])
			self.diffs.append(diff)

		print(f'diffs: {[round(a, 2) for a in self.diffs]}')

		for i in range(size):
			self.diffs[i] = self.diffs[i] / (self.data[i] + 1)
			# self.diffs[i] = self.data[i] / self.diffs[i]

		print(f'percentage diffs: {[round(a, 2) for a in self.diffs]}')

		for i in range(size):
			if self.diffs[i] > 0.4:
				print()
			
			print(f'{self.data[i]}, ', end = '')
			
		return self.diffs
	
	def near(self, delta):
		return -64 < delta < 65

	def iterative_joining(self):

		stds = [self.inf] * len(self.indexes)

		index = self.indexes[0]
		groups = [[]]

		# Continue until best chaining is found
		while len(self.indexes) > 0:

			section = self.clusters[index]
			place = self.indexes.index(index)
			self.indexes.remove(index)
			groups[-1].append(index)

			# print(f'{index}: {section} | rem: {self.indexes}')

			metric = False

			# Look at all next thingies
			# for i in self.indexes:
			for i in range(len(self.indexes)):

				next_index = self.indexes[i]
				diff = next_index - index

				if diff < -64:
				# if diff < 0:
					continue

				if diff > 65:
				# if diff > 128:
					metric = False
					break

				next_section = self.clusters[i]

				metric = self.near(section[-1] - next_section[0])

				# if not metric and False:

				# 	A = stds[index]
				# 	if A == self.inf:
				# 		A = stds[index] = np.std(section)
				# 	B = stds[i]
				# 	if B == self.inf:
				# 		B = stds[i] = np.std(next_section)

				# 	T = np.std(section[self.block_size // 2:] + next_section[:self.block_size // 2])
				# 	metric = (A + B) > (T)

				# 	# if metric:
				# 	# 	print(f'{section} -> {next_section} : {round(A + B)} {round(T)} {metric}')

				if metric:
					# if i - index > 1:
					# 	print(f'JUMP {i - index} BLOCKS')
					index = next_index
					break
		
			if not metric:
				# print(f'POP (doesnt matter relative order)')
				if len(self.indexes) > 0:
					groups.append([])
					print(f'place: {next_index} : len: {len(self.indexes)}')
					index = self.indexes[next_index]

		groups.sort(key = lambda a: a[-1])
		# print([f'{group[0]} - {group[-1]}' for group in groups])

		self.order = list(chain.from_iterable(groups))

		# print(self.order)

		# out = [self.clusters[i] for i in order]

		out = []
		for i in self.order:
			out += self.clusters[i]

		# print(out[:1000])

		return out
	
	def get_group_jumps(self):

		buffer = self.indexes_copy
		indexes = self.order

		prev_idx = -1
		idx = 0

		jumps = {}

		i = 0
		while len(indexes) > 0:

			prev_idx = idx
			idx = indexes.pop(i)

			diff = idx - prev_idx

			if diff != 1 and -33 < diff < 32:
				jumps[prev_idx] = diff
				# print(f'at {prev_idx:2} jump {diff:3} ')
			else:
				# print(f'run {idx}')
				pass

			# i += 1

		# print(jumps)

		return jumps

	def initial_cluster(self):

		# self.clusters = [[i] for i in self.data]
		self.clusters = [self.data[i : i + self.block_size] for i in range(0, len(self.data), self.block_size)]

		self.indexes = list(range(len(self.clusters)))
		self.indexes_copy = self.indexes.copy()

		return self.clusters

		"""
		for i in range(len(self.clusters) - 1, 0, -1):

			prev = self.clusters[i - 1]
			curr = self.clusters[i]

			delta = curr[-1] - prev[0]

			if self.near(delta):
				self.clusters[i] = self.clusters[i - 1] + self.clusters[i]
				self.clusters.pop(i - 1)

		# self.indexes = [i for i in range(len(self.clusters))]
		self.indexes = list(range(len(self.clusters)))
		self.indexes_copy = self.indexes.copy()

		print(self.clusters)
		print(self.indexes)
		"""

if __name__ == '__main__':

	import random
	random.seed(0)

	# data = [0, 0, 1, 1, 0, 1, 0, 0]
	data = [10, 0, 0, 31, 255, 255, 254, 230, 10, 11, 0, 2, 234, 219, 100, 50] * 2
	# data = [1.91, 2.87, 3.61, 10.91, 11.91, 12.82, 100.73, 100.71, 101.89, 200]
	
	print(data)

	p = Partitioner(data, block_size = 4)
	p.initial_cluster()
	groups = p.iterative_joining()

	print(groups)
	p.get_group_jumps(groups)
	# a = p.cluster_percentage()

	# print(splits)

	exit()

	# data = [random.randint(0, 255) for i in range(20)]

	data = [[i] for i in data]

	# print(data)

	p = Partitioner(data)
	p.initial_cluster()

	# p.iterative_joining()

	it = 0

	# while len(data) > 1:
	# 	# p.join_adj()
	# 	p.join_all()
	# 	it += 1
	# 	# print(it, len(data))
	
	# print(p.indexes)

	# ps = p.set_prefix_sum(data)
	# print(ps)

	# for i in range(0, len(data)):
	# 	d = p.get_mean_half_diff(i)

	# 	print(f'split = {i} : {d}')