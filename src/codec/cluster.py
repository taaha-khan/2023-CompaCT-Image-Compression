
from scipy.stats import entropy
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
		return -65 < delta < 64

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

				if diff < -65:
				# if diff < 0:
					continue

				if diff > 64:
				# if diff > 128:
					metric = False
					break

				next_section = self.clusters[i]

				# metric = self.near(section[-1] - next_section[0])

				if not metric or True:

					A = stds[index]
					if A == self.inf:
						A = stds[index] = np.std(section)
					B = stds[i]
					if B == self.inf:
						B = stds[i] = np.std(next_section)

					T = np.std(section[self.block_size // 2:] + next_section[:self.block_size // 2])
					metric = (A + B) > (T)

					# if metric:
					# 	print(f'{section} -> {next_section} : {round(A + B)} {round(T)} {metric}')

				if metric:
					# if i - index > 1:
					# 	print(f'JUMP {i - index} BLOCKS')
					index = next_index
					break
		
			if not metric:
				# print(f'POP (doesnt matter relative order)')
				if len(self.indexes) > 0:
					groups.append([])
					# print(f'place: {next_index} : len: {len(self.indexes)}')
					index = self.indexes[0]

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

			if diff != 1 and -65 < diff < 64:
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

class BlockPartitioner:

	# TODO: Minimize memory with smaller datatypes

	def __init__(self, data = None, order = None, block_size = 16):
		self.data = data
		self.order = order
		self.block_size = block_size
		self.size = len(data)

	def get_entropy(self, data):
		value, counts = np.unique(data, return_counts = True)
		return entropy(counts)

	def initial_partition(self):

		array = np.asarray(self.data, dtype = np.int32)
		order = np.asarray(self.order, dtype = np.int32)
		
		self.blocks       = array.reshape((self.size // self.block_size, self.block_size))
		self.block_orders = order.reshape((self.size // self.block_size, self.block_size))

		return self.blocks

	def set_delta_changes_array(self):
		""" O(N) Get prefix sum array of immediate changes """

		self.prefix_sum = np.zeros(self.size, dtype = np.uint32)

		for i in range(1, self.size):

			# Count number of times can't use delta encoding
			diff = abs(self.data[i] - self.data[i - 1])
			change_delta = -63 > diff or diff > 64
			
			self.prefix_sum[i] = int(self.prefix_sum[i - 1] + int(change_delta))
		
		return self.prefix_sum

	def get_num_delta_changes(self, l, r):
		""" O(1) Get changes given subsection inclusive """
		return self.prefix_sum[r] - self.prefix_sum[l]

	def block_partition(self):

		block_deltas = {}
		for i, block in enumerate(self.blocks):

			start = i * self.block_size
			end = start + self.block_size - 1
			
			changes = self.get_num_delta_changes(start, end)
			if changes >= self.block_size / 2: # FIXME: Cutoff
				block_deltas[i] = changes
		
		# print('block_deltas', block_deltas)

		# FIXME: Remove or actually use
		# difficult_blocks = sorted(block_deltas, key = block_deltas.get, reverse = True)
		# print('difficult_blocks', len(difficult_blocks))

		completed = np.full(self.size // self.block_size, False)

		output = []

		PIXEL_ORDER = np.zeros(self.size, dtype = np.int32)
		BLOCK_JUMPS = {}

		running_index = 0

		num_skips = 0

		# O(num_blocks)
		for i, block in enumerate(self.blocks):	
			
			start_index = i * self.block_size
			ended_index = start_index + self.block_size - 1

			# og = np.asarray(self.data[start_index : ended_index])
			# og_d = og[1:] - og[:-1]
			# og_entropy = self.get_entropy(og_d)

			# Block doesn't need help
			if i not in block_deltas and not completed[i]:
				# print(f'running_index: [{running_index} : {running_index + self.block_size}] | {self.block_orders[i]}')
				
				PIXEL_ORDER[running_index : running_index + self.block_size] = self.block_orders[i]
				running_index += self.block_size

				completed[i] = True
				continue

			if completed[i]:
				continue

			# Current encoding scheme will result in N delta changes
			# TODO: Add previous block's last pixel possible large delta

			# Look at default next block
			next_i = i + 1
			while completed[i] and next_i < self.size - 1:
				next_i += 1
			
			ended_index = next_i * self.block_size - 1
			current_delta = self.get_num_delta_changes(start_index - 1, ended_index)

			# print()
			# print(f'current_delta from block {i} to {next_i}:', current_delta)

			# Block needs help: preview future
			# O(jump_size = 64)

			meshed = False

			# FIXME: Iterate future blocks that aren't completed
			# enumerate(self.blocks[next_i + 1 : next_i + 64])
			for j, next_block in enumerate(self.blocks[i + 1 : i + 64]):

				next_index = i + j + 1

				# TODO: Maybe also only look at next blocks that also need help
				# If next block has already been completed
				if completed[next_index]:
					continue
				
				A = block
				B = next_block

				# print('A', A)
				# print('B', B)

				# Interleaving A and B
				C = np.empty((2 * self.block_size), dtype = A.dtype)
				C[0::2] = A
				C[1::2] = B
				# print('C', C)

				# Differences between neighboring pixels
				D = C[1:] - C[:-1]
				# print('D', D)

				# new_entropy = self.get_entropy(D)

				# print(f'OG: {og} : E: {og_entropy}')
				# print(f'DG: {D} : E: {new_entropy}')
				# print()
				
				num_changes = np.count_nonzero((-64 <= D) & (D >= 65)) + 1
				# print('num_changes', num_changes)

				# FIXME: Cutoff
				# if num_changes / new_entropy < (current_delta - 2) / og_entropy:
				if num_changes < current_delta - 2:
				# if new_entropy < og_entropy:

					meshed = True

					num_skips += 1
					# print(f'mesh {i} + {next_index - i} = {next_index}')

					BLOCK_JUMPS[i] = next_index
					
					# JUMP
					completed[i] = True
					completed[next_index] = True

					# FIXME: Looks gross
					PIXEL_ORDER[running_index : running_index + 2 * self.block_size : 2] = self.block_orders[i]
					PIXEL_ORDER[running_index + 1 : running_index + 2 * self.block_size + 1 : 2] = self.block_orders[next_index]
					
					# print(PIXEL_ORDER[running_index - 3 : running_index + 32 + 3])

					running_index += 2 * self.block_size

					# TODO: Look at best in preview or look at first efficient?
					break

					# TODO: Continue off after latest mesh instead of popping back?

			# Couldn't find help
			if not meshed:
				# print(f'running_index: [{running_index} : {running_index + self.block_size}] | {self.block_orders[i]}')
				PIXEL_ORDER[running_index : running_index + self.block_size] = self.block_orders[i]
				running_index += self.block_size
				completed[i] = True

		# print(f'num_skips: {num_skips}')

		# all_orders = set(np.arange(self.size))
		# got_orders = set(PIXEL_ORDER)

		# print(f'incomplete: {all_orders - got_orders}')

		return (PIXEL_ORDER, BLOCK_JUMPS)

if __name__ == '__main__':

	import random
	random.seed(0)

	# data = [0, 0, 1, 1, 0, 1, 0, 0]
	# data = [10, 0, 0, 31, 255, 255, 254, 230, 10, 11, 0, 2, 234, 219, 100, 50] * 2
	# data = [1.91, 2.87, 3.61, 10.91, 11.91, 12.82, 100.73, 100.71, 101.89, 200]
	
	data = ([0, 100, 200, 300] + [50, 150, 250, 350]) # * 8 # 32768
	# data = [random.randrange(4096) for i in range(64)]
	print('data', data)
	
	order = range(len(data))
	order = [0, 2, 4, 6, 1, 3, 5, 7]
	d2 = [data[i] for i in order]
	print('d2', d2)

	data = d2

	p = BlockPartitioner(data, order = order, block_size = 4)
	d = p.set_delta_changes_array()
	# print('delta changes', d)

	i = p.initial_partition()
	print(i)

	a = p.block_partition()
	print(a)

	# p = Partitioner(data, block_size = 4)
	# p.initial_cluster()
	# groups = p.iterative_joining()

	# print(groups)
	# p.get_group_jumps(groups)
	# a = p.cluster_percentage()

	# print(splits)
