
import numpy as np

w = 101
h = 92

a = np.arange(w * h).reshape(h, w)

def index(p):
	r, c = p
	return r * w + c

print(a)

for c in range(w):
	for r in range(h):
		p = (r, c)
		print(p, a[p] / index(p))
