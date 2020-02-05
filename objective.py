from math import sqrt

def f(radiuses):
	rCur = 0
	rMax = radiuses[0]
	indMax = 0
	width = radiuses[0]
	distance = 0
	for i in range(1, len(radiuses)):
		for j in range(indMax, i):
			distance = 2 * sqrt(radiuses[j+1] * rMax)

		if radiuses[i] > rMax or rMax < distance + radiuses[i]:
			width += distance
			rMax = radiuses[i]
			indMax = i
	width += radiuses[-1]
	return width