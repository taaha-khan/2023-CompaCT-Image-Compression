
from scipy.stats import entropy
from itertools import chain
import numpy as np

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
