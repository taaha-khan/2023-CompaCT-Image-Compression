
import numpy as np

class Partitioner:
	
	def __init__(self, data, min_block_size = float('-inf')):

		self.data = data
		self.size = len(data)

		self.min_block_size = min_block_size

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

	def get_mean_half_diff(self, p):

		if p >= self.size - 1:
			return float('inf')

		if p < self.min_block_size or self.size - p - 2 < self.min_block_size:
			return float('inf')

		l = self.get_diff(0, p)
		r = self.get_diff(p + 1, self.size - 1)

		return (l + r) / 2

	def join_adj(self):

		min_delta = float('inf')
		min_pair = None

		for i in range(len(self.data) - 1):

			a = self.data[i]
			b = self.data[i + 1]

			joined = a + b

			delta = (max(joined) - min(joined)) # / len(joined)

			# print(f'{i} - {i + 1} : {delta}')

			if delta < min_delta:
				min_delta = delta
				min_pair = (i, i + 1)

		self.data[min_pair[0]] = self.data[min_pair[0]] + self.data[min_pair[1]]
		self.data.pop(min_pair[1])

		print(self.data, min_pair)


	def join_all(self):

		min_delta = float('inf')
		min_pair = None

		joins = []

		n = 0

		for i in range(len(self.data) - 1):

			for j in range(i + 1, len(self.data)):

				n += 1

				a = self.data[i]
				b = self.data[j]

				joined = a + b

				# delta = (max(joined) - min(joined)) # / len(joined)
				delta = abs(joined[0] - joined[-1]) # * (j - i)

				# print(f'{i} - {i + 1} : {delta}')

				if delta < min_delta:
					min_delta = delta
					min_pair = (i, j)

		self.data[min_pair[0]] = self.data[min_pair[0]] + self.data[min_pair[1]]
		self.data.pop(min_pair[1])

		self.indexes[min_pair[0]] = self.indexes[min_pair[0]] + self.indexes[min_pair[1]]
		self.indexes.pop(min_pair[1])

		joins.append(min_pair)

		print(self.data, min_pair)
		# print(joins)

		return
	
	def near(self, delta):
		return -32 < delta < 33

	def iterative_joining(self):

		first = 0

		# Continue until best chaining is found
		while True:

			section = data[first]

			# Look at all next thingies
			for i in range(first + 1, len(data)):
				next_section = data[i]

				if self.near(section[-1] - next_section[0]):
					print(f'JUMP TO INDEX {i}')
					first = i
					break
		
			if i == len(data) - 1:
				pass

	def initial_cluster(self):

		self.clusters = self.data.copy()

		for i in range(len(self.clusters) - 1, 0, -1):

			prev = self.clusters[i - 1]
			curr = self.clusters[i]

			delta = curr[-1] - prev[0]

			if self.near(delta):
				self.clusters[i] = self.clusters[i - 1] + self.clusters[i]
				self.clusters.pop(i - 1)

		self.indexes = [[i] for i in range(len(self.clusters))]

		print(self.clusters)
		print(self.indexes)


if __name__ == '__main__':

	import random
	random.seed(0)

	# data = [0, 0, 1, 1, 0, 1, 0, 0]
	data = [10, 0, 0, 32, 255, 255, 254, 230, 10, 11, 0, 2, 234, 219, 50, 100, 101] # * 100

	print(data)

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