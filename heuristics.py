from math import sqrt, ceil, factorial, e
import numpy as np
import time
import random
import re
from copy import deepcopy

# Own modules
import permutation_generator
from objective import f
import rep_solution
import problem

MAX_INT = 1.7976931348623157e+308
MIN_INT = 2.2250738585072014e-308

def printUnderscoreNewElement():
	print("  |")
	print(" (+)")
	print("   \____________________________________________________________________________________"
		"__________________________________________")

def printUnderscoreOuterMost():
	print("   _____________________________________________________________________________________"
		"__________________________________________")

def printUnderscoreOuter():
	print("    |___________________________________________________________________________________"
		"__________________________________________")

def printDashInner():
	print("         -------------------------------------------------------------------------------"
		"------------------------------------------")

def format3Dec(number):
	return "{:.3f}".format(number)

def format8Dec(number):
	return "{:.8f}".format(number)

def beforeRunMessage(problemName, numTests, inputSize):
	printUnderscoreOuterMost()
	print("  |"+problemName)
	print("  |Number of tests is", numTests, "and problem set consists of", inputSize, "circles.")
	print("  |Trying to minimize optimal value.")
	print("  |Efficiency = initialCost / optimalCost")

def heuristicDescription(heuristicName, info=""):
	printUnderscoreNewElement()
	print("   ", "|"+heuristicName, info)
	printUnderscoreOuter()

def afterTestMessage(test, optimal, initial, runTime, efficiency):
	print("   \t|TEST", test, "\t: Optimal =", optimal,
				 "\tInitial =", initial,
				 "\tRunning Time =", runTime, "seconds", "\tEfficiency =", efficiency)

def afterTestMessageV2(test, optimal, runTime):
	print("   \t|TEST", test, "\t: Optimal =", optimal, "\tRunning Time =", runTime, "seconds")    

def afterRunMessage(runTime):
	printDashInner()
	print("   \t\t\t| Average Running Time:", runTime, "seconds |")

def formatProblemList(problems):
	if not any(isinstance(i, list) for i in problems):
		lst = list()
		lst.append(problems)
		return lst
	return problems

def generateTestSet(N, testNum):
	return np.ndarray.tolist(np.random.randint(1, 1000, size = (testNum, N)))

def generateNeighbors(solution, kLimit):
	neighbors = []
	length = len(solution)
	for k in range(1, kLimit):
		for i in range(length):
			neighbor = deepcopy(solution)
			temp = neighbor[(i+k)%length]
			neighbor[(i+k)%length] = neighbor[i]
			neighbor[i] = temp
			neighbors.append(neighbor)
	return neighbors

def getFitnessesOfNeighbours(neighbourhood):
	result = list()
	for neighbour in neighbourhood:
		result.append(f(neighbour))
	return result

def getWorstIndexes(population, elitismLevel):
	if elitismLevel <= 0:
		return []
	worstList = [MIN_INT] * elitismLevel
	worstIndexes = [MIN_INT] * elitismLevel
	for i in range(len(population)):
		temp = f(population[i])        
		maximum = min(worstList)
		if temp > maximum:
			curInd = worstList.index(min(worstList))
			worstList[curInd] = temp
			worstIndexes[curInd] = i
	return worstIndexes

def getFittestIndexes(population, elitismLevel):
	if elitismLevel <= 0:
		return []
	fittestList = [MAX_INT] * elitismLevel
	fittestIndexes = [-1] * elitismLevel
	for i in range(len(population)):
		temp = f(population[i])        
		minimum = max(fittestList)
		if temp < minimum:
			curInd = fittestList.index(max(fittestList))
			fittestList[curInd] = temp
			fittestIndexes[curInd] = i
	return fittestIndexes
	
def getFittestIndividuals(population, populationPerm, elitismLevel):
	if elitismLevel <= 0:
		return []
	fittestList = [MAX_INT] * elitismLevel
	fittestSol = [None] * elitismLevel
	fittestSolPerm = [None] * elitismLevel
	for i in range(len(population)):
		temp = f(population[i])        
		minimum = max(fittestList)
		if temp < minimum:
			curInd = fittestList.index(max(fittestList))
			fittestList[curInd] = temp
			fittestSol[curInd] = deepcopy(population[i])
			fittestSolPerm[curInd] = deepcopy(populationPerm[i])
	return fittestSol, fittestSolPerm


def rouletteSelection(population, populationPerm, POPULATION_SIZE, minimize=True):
	'''
		Roulette Wheel parentSelection Method
	'''
	# Calculate Fitness
	fitnessValues = getFitnessesOfNeighbours(population)
	probabilities = [None] * len(population)
	totalFitness = sum(fitnessValues)

	# Generate probabilities
	fitnessValues.sort()
	if minimize:
		divider = pow(POPULATION_SIZE, 2)
		for i in range(len(probabilities)):
			probabilities[i] = (totalFitness / fitnessValues[i]) / divider
		probabilities.sort(reverse = True)
	else:
		for i in range(len(probabilities)):
			probabilities[i] = fitnessValues[i] / totalFitness
		probabilities.sort()

	# Select remaining parents with roulette wheel selection.
	for i in range(len(population)):
		randProbMagnitude = random.random()
		if randProbMagnitude < probabilities[0]:
			population[i] = population[0]
			populationPerm[i] = populationPerm[0]
		else:
			predecessorSum = 0.0
			for j in range(1, len(probabilities)):
				predecessorSum += probabilities[j-1]
				if (randProbMagnitude >= predecessorSum) and (randProbMagnitude < predecessorSum + probabilities[j]):
					population[i] = population[j]
					populationPerm[i] = populationPerm[j]
					break;

def tournamentSelection(population, populationPerm, minimize=True, k = 2):
	selected = 0
	newPopulation = []
	newPopulationPerm = []
	
	while selected != len(population):
		indexes = random.sample(range(0, len(population)), k)
		minimum = MAX_INT
		minimumInd = -1
		for ind in indexes:
			temp = f(population[ind])
			if temp < minimum:
				minimum = temp
				minimumInd = ind
		newPopulation.append(population[minimumInd])
		newPopulationPerm.append(populationPerm[minimumInd])
		selected += 1
	population = newPopulation
	populationPerm = newPopulationPerm

def tournamentSelectionStoc(population, populationPerm, minimize=True, k = 2):
	selected = 0
	newPopulation = []
	newPopulationPerm = []
	magnitude = 0.95
	while selected != len(population):
		probability = random.uniform(0, 1)
		if probability <= magnitude:
			indexes = random.sample(range(0, len(population)), k)
			minimum = MAX_INT
			minimumInd = -1
			for ind in indexes:
				temp = f(population[ind])
				if temp < minimum:
					minimum = temp
					minimumInd = ind
			newPopulation.append(population[minimumInd])
			newPopulationPerm.append(populationPerm[minimumInd])
			magnitude = magnitude * (pow((1 - magnitude), selected))
			selected += 1
	population = newPopulation
	populationPerm = newPopulationPerm

def survivorSelection(
	population, 
	populationPerm, 
	fittestParents, 
	fittestParentsPerm, 
	elitismLevel):
	worstIndexes = getWorstIndexes(population, elitismLevel)
	for i in range(elitismLevel):
		population[worstIndexes[i]] = fittestParents[i]
		populationPerm[worstIndexes[i]] = fittestParentsPerm[i]

def shuffleMatingPool(population, populationPerm):
	indexes = random.sample(range(0, len(population)), len(population))
	newPopulation = []
	newPopulationPerm = []
	for i in range(len(population)):
		newPopulation.append(population[indexes[i]])
		newPopulationPerm.append(populationPerm[indexes[i]])
	population = newPopulation
	populationPerm = newPopulationPerm

def findStartIndex(perm1, perm2):
	for i in range(len(perm1)):
		if perm1[i] != perm2[i]:
			return i

def createCycle(index, perm1, perm2):
	cycle = [-1] * len(perm1)
	cycle[index] = index
	last = -1
	first = perm1[index]
	elementsInCycle = 1 
	
	while last != first:
		last = perm2[index]
		if last == first:
			break
		index = perm1.index(last)
		cycle[index] = index
		elementsInCycle += 1
	return cycle, elementsInCycle


def cycleXover(parent1, parent2, parent1Perm, parent2Perm):
	'''
		Cycle Crossover
	'''
	if parent1 == parent2:
		return parent1, parent2, parent1Perm, parent2Perm

	start = findStartIndex(parent1Perm, parent2Perm)
	cycle, elementsInCycle = createCycle(start, parent1Perm, parent2Perm)
	child1 = deepcopy(parent1)
	child1Perm = deepcopy(parent1Perm)
	child2 = deepcopy(parent2)
	child2Perm = deepcopy(parent2Perm)
	size = len(cycle)
	if (elementsInCycle > 2 and size > 5) or size < 5:
		for i in range(size):
			if cycle[i] == -1:
				temp = child1[i]
				child1[i] = child2[i]
				child2[i] = temp
				temp = child1Perm[i]
				child1Perm[i] = child2Perm[i]
				child2Perm[i] = temp
	else:
		return [], [], [], []
	# Other elements in first cycle remain same for both parent
	return child1, child2, child1Perm, child2Perm

def order1Generate(parent1, parent2, parent1Perm, parent2Perm, start, end):
	size = len(parent1)
	child = [None] * size
	child[start:end] = parent1[start:end]
	childPerm = [None] * size
	childPerm[start:end] = parent1Perm[start:end]
	inserted = end - start
	i = end % size
	j = end % size
	while inserted < size:
		if parent2Perm[i] not in childPerm:
			child[j] = parent2[i]
			childPerm[j] = parent2Perm[i]
			inserted += 1
			j += 1
			j %= size
		i += 1
		i %= size
	return child, childPerm


def order1Xover(parent1, parent2, parent1Perm, parent2Perm):
	'''
		order1 crossover
	'''
	if parent1 == parent2:
		return parent1, parent2, parent1Perm, parent2Perm
	indexes = random.sample(range(0, len(parent1)), 2)
	indexes.sort()
	child1, child1Perm = order1Generate(parent1, parent2, parent1Perm, parent2Perm, indexes[0], indexes[1])
	child2, child2Perm = order1Generate(parent2, parent1, parent2Perm, parent1Perm, indexes[0], indexes[1])
	return child1, child2, child1Perm, child2Perm

def crossover(population, populationPerm):
	# Pc is generally between (0.6-0.9).
	crossProbMagnitude = 0.85;
	pairNum = len(population) // 2
	# feasibility check
	for i in range(pairNum):
		probability = random.uniform(0, 1)
		if probability < crossProbMagnitude:
			temp1, temp2, temp1Perm, temp2Perm = cycleXover(population[i*2], population[i*2+1], populationPerm[i*2], populationPerm[i*2+1])
			if temp1 == []:
				temp1, temp2, temp1Perm, temp2Perm = order1Xover(population[i*2], population[i*2+1], populationPerm[i*2], populationPerm[i*2+1])
			if temp1 != population[i*2] and temp2 != population[i*2+1]:
				population[i*2] = temp1
				population[i*2+1] = temp2
				populationPerm[i*2] = temp1Perm
				populationPerm[i*2+1] = temp2Perm

def insertMutation(solution, solutionPerm):
	'''
		Insert Mutation
	'''
	ind1, ind2 = random.sample(range(0, len(solution)), 2)
	end = max(ind1, ind2)
	start = min(ind1, ind2)
	
	if abs(ind1 - ind2) == 1:
		if end == len(solution) - 1:
			start -= 1
		else:
			end += 1    
	temp = solution[end]
	tempPerm = solutionPerm[end]

	for i in reversed(range(start+1, end)):
		solution[i+1] = solution[i]
		solutionPerm[i+1] = solutionPerm[i]
	solution[start+1] = temp
	solutionPerm[start+1] = tempPerm

def swapMutation(solution, solutionPerm):
	'''
		Swap Mutation
	'''
	ind1, ind2 = random.sample(range(0, len(solution)), 2)

	temp = solution[ind1]
	solution[ind1] = solution[ind2]
	solution[ind2] = temp

	temp = solutionPerm[ind1]
	solutionPerm[ind1] = solutionPerm[ind2]
	solutionPerm[ind2] = temp

def inverseMutationSingle(solution, solutionPerm):
	'''
		Inverse Mutation
	'''
	indexes = random.sample(range(0, len(solution)), 2)
	indexes.sort()
	temp = solution[:indexes[1]+1]
	tempPerm = solutionPerm[:indexes[1]+1]
	temp.reverse()
	tempPerm.reverse()
	if indexes[0] != 0:
		temp = temp[:-indexes[0]]
		tempPerm = tempPerm[:-indexes[0]]
	solution[indexes[0]:indexes[1]+1] = temp
	solutionPerm[indexes[0]:indexes[1]+1] = tempPerm

def inverseMutation(solution, solutionPerm, k=2):
	indexes = random.sample(range(0, len(solution)), k)
	indexes.sort()
	for i in range(len(indexes)//2):
		end = indexes[i*2+1]
		start = indexes[i*2]
		temp = solution[:end+1]
		tempPerm = solutionPerm[:end+1]
		temp.reverse()
		tempPerm.reverse()
		if start != 0:
			temp = temp[:-start]
			tempPerm = tempPerm[:-start]
		solution[start:end+1] = temp
		solutionPerm[start:end+1] = tempPerm

def scrambleMutation(solution, solutionPerm, k=2):
	indexes = random.sample(range(0, len(solution)), k)
	indexes.sort()
	for i in range(len(indexes)//2):
		end = indexes[i*2+1]
		start = indexes[i*2]
		temp = solution[start:end+1]
		tempPerm = solutionPerm[start:end+1]
		positions = random.sample(range(0, len(temp)), len(temp))
		newTemp = []
		newTempPerm = []
		for ind in positions:
			newTemp.append(temp[ind])
			newTempPerm.append(tempPerm[ind])
		solution[start:end+1] = newTemp
		solutionPerm[start:end+1] = newTempPerm        

def mutation(population, populationPerm, iteration, MAX_ITERATION):
	magnitude = 0.15
	for i in range(len(population)):
		probability = random.uniform(0, 1)
		if probability <= magnitude:
			if iteration % 25 == 0:
				scrambleMutation(population[i], populationPerm[i])
			elif iteration % 5 == 0:
				inverseMutation(population[i], populationPerm[i])
			else:
				swapMutation(population[i], populationPerm[i])  

def getIndexPositions(listOfElements, element):
	'''
		Find all occurences of an element in a list.
	'''
	indexPosList = []
	indexPos = 0
	while True:
		try:
			# Search for item in list from indexPos to the end of list
			indexPos = listOfElements.index(element, indexPos)
			# Add the index position in list
			indexPosList.append(indexPos)
			indexPos += 1
		except ValueError as e:
			break
	return indexPosList

def countUniques(lst):
	temp = []
	count = 0
	for el in lst:
		if el not in temp:
			temp.append(el)
			count += 1 
	return count

def initialPopulation(solution, permutation, POPULATION_SIZE):
	neighbourhood = list()
	populationPerm = list()
	size = len(solution)
	while len(neighbourhood) < POPULATION_SIZE:
		newSol = random.sample(solution, size)
		if newSol not in neighbourhood:
			neighbourhood.append(newSol)
			newPerm = []
			for element in newSol:
				count = solution.count(element)
				if count == 1:
					newPerm.append(solution.index(element) + 1)
				else:
					allOccurences = getIndexPositions(solution, element)
					for ind in allOccurences:
						if ind + 1 not in newPerm:
							newPerm.append(ind + 1)
							break    
			populationPerm.append(newPerm)
	return neighbourhood, populationPerm

def bruteForce():
	problem = rep_solution.generateRandomSolution()
	size = len(problem)
	permutations = permutation_generator.generator(size)
	optimalCost = MAX_INT
	start = time.time()
	for permutation in permutations:
		arrangement = []
		for j in range(0, size):
			arrangement.append(problem[permutation[j]-1])
		result = f(arrangement)
		if result < optimalCost:
			optimalCost = result
	return format3Dec(optimalCost), "........", "...."

def geneticAlgorithm(MAX_ITERATION, POPULATION_SIZE=100):
	'''Genetic Algorithm'''
	optimal = rep_solution.generateRandomSolution()
	optimalCost = f(optimal)
	initialCost = optimalCost
	size = len(optimal)
	initialPermutation = [i for i in range(1, size+1)]

	'''
	if size < 20:
		MAX_ITERATION = size * 3
	else:
		MAX_ITERATION = (size * 3) // 2
	'''

	if size > 100:
		POPULATION_SIZE = size
	else:
		# Set population size
		numUnique = countUniques(optimal)
		if numUnique < 5:
			POPULATION_SIZE = factorial(numUnique)
	elitismLevel = POPULATION_SIZE // 2
	
	# Create initial population
	population, populationPerm = initialPopulation(optimal, initialPermutation, POPULATION_SIZE)
	for j in range(MAX_ITERATION):
		tournamentSelection(population, populationPerm, POPULATION_SIZE, k=2)
		shuffleMatingPool(population, populationPerm)
		fittestParents, fittestParentsPerm = getFittestIndividuals(population, populationPerm, elitismLevel)
		crossover(population, populationPerm)
		mutation(population, populationPerm, j, MAX_ITERATION)
		survivorSelection(population, populationPerm, fittestParents, fittestParentsPerm, elitismLevel)
		minimum = min(getFitnessesOfNeighbours(population))
		if minimum < optimalCost:
			optimalCost = minimum
	return format3Dec(optimalCost), format3Dec(initialCost), format3Dec(initialCost / optimalCost)

def iteratedLocalSearch(kLimit = 4, MAX_ITERATION = 3):
	optimalCost = MAX_INT
	for j in range(MAX_ITERATION):
		found = False
		localOptimal = rep_solution.generateRandomSolution()
		size = len(localOptimal)
		localOptimalCost = f(localOptimal)
		initialCost = localOptimalCost
		generation = 0
		while found is False:
			newLimit = (ceil(kLimit - generation / size))
			if newLimit <= 2:
				newLimit = 2
			generation += 4

			neighbourhood = generateNeighbors(localOptimal, newLimit)
			neighbourhoodWidhts = getFitnessesOfNeighbours(neighbourhood)
			minNeighbour = min(neighbourhoodWidhts)
			if localOptimalCost <= minNeighbour:
				found = True
			else:
				localOptimal = neighbourhood[neighbourhoodWidhts.index(minNeighbour)]
				localOptimalCost = f(localOptimal)
		if localOptimalCost < optimalCost:
			optimalCost = localOptimalCost
	return format3Dec(optimalCost), format3Dec(initialCost), format3Dec(initialCost / optimalCost)

def tabuSearch(kLimit=4):
	MAX_ITER = 300
	TABU_TENURE = 7
	curIterNum = 0
	tanures = []
	tabuList = {}
	optimal = rep_solution.generateRandomSolution()
	size = len(optimal)
	initialCost = f(optimal)
	optimalCost = initialCost
	optimalCandidate = optimal
	tabuList[tuple(optimal)] = TABU_TENURE
	generation = 0    
	while curIterNum != MAX_ITER:
		newLimit = (ceil(kLimit - generation / size))
		if newLimit <= 2:
			newLimit = 2
		neighbourhood = generateNeighbors(optimalCandidate, newLimit)
		neighbourhood = random.sample(neighbourhood, len(neighbourhood))
		optimalCandidate = neighbourhood[0]

		for candidate in neighbourhood:
			if (tuple(candidate) not in tabuList) and (f(candidate) < f(optimalCandidate)):
				optimalCandidate = candidate
		
		candidateCost = f(optimalCandidate)
		if candidateCost < optimalCost:
			optimalCost = candidateCost
			optimal = optimalCandidate

		for key in tabuList.keys():
			tabuList[key] -= 1

		if tuple(optimalCandidate) not in tabuList:
			if len(tabuList) > 50:
				for i in range(10):
					del tabuList[min(tabuList, key=tabuList.get)]
			tabuList[tuple(optimalCandidate)] = TABU_TENURE
		generation += 4
		curIterNum += 1
	return format3Dec(optimalCost), format3Dec(initialCost), format3Dec(initialCost / optimalCost)


def simulatedAnnealing():
	'''
		Vc = current solution
		Vn = new solution
		T = temperature
	'''
	Tmax = 100
	Tmin = pow(10, -8)
	iterMax = 5000
	T = Tmax
	coolingRatio = 0.98
	i = 1
	Vc = rep_solution.generateRandomSolution()
	Vc_cost = f(Vc)
	optimal = Vc
	optimalCost = Vc_cost
	initialCost = Vc_cost
	trialForOptimum = 0
	kLimit = len(optimal)//2+1
	while i <= iterMax or T > Tmin:
		if trialForOptimum >= iterMax // 3:
			trialForOptimum = 0
			restart = True
		if T <= Tmin and restart is True:
			Vc = rep_solution.generateRandomSolution()   
			Vc_cost = f(Vc)
			T = Tmax
		
		neighbors = generateNeighbors(Vc, kLimit)
		if T <= Tmin and restart is False:
			allNeighborsCosts = getFitnessesOfNeighbours(neighbors)
			bestCost = MAX_INT
			for j in range(len(allNeighborsCosts)):
				if bestCost > allNeighborsCosts[j]:
					bestCost = allNeighborsCosts[j]
				Vn = neighbors[allNeighborsCosts.index(bestCost)]
				Vn_cost = bestCost
		else:
			restart = False
			ind = random.randint(0, len(neighbors)-1)
			Vn = neighbors[ind]
			Vn_cost = f(Vn)
		deltaCost = Vn_cost - Vc_cost 
		# Operation on cooled state.
		if deltaCost < 0: 
			Vc = Vn
			Vc_cost = Vn_cost
			if Vc_cost < optimalCost:
				optimal = Vn
				optimalCost = Vn_cost
				trialForOptimum = 0
		elif deltaCost / T < 1 and random.random() >= pow(e, deltaCost / T):
			Vc = Vn
			Vc_cost = Vn_cost
			if Vc_cost < optimalCost:
				optimal = Vn
				optimalCost = Vn_cost
				trialForOptimum = 0
		elif T <= Tmin and restart is False:
			Vc = rep_solution.generateRandomSolution()
			Vc_cost = f(Vc)
		T *= coolingRatio
		i += 1
		trialForOptimum += 1
		if i == iterMax:
			break
	return format3Dec(optimalCost), format3Dec(initialCost), format3Dec(initialCost / optimalCost)

def neighborsProbs(neighbors_costs, D):
	probs = []
	for cost in neighbors_costs:
		probs.append(cost / D)
	return probs    

def stochasticVNS():
	solution = rep_solution.initialSolution()
	solutionCost = f(solution)
	optimal, optimalCost = solution[:], solutionCost
	initialCost = solutionCost
	stop = False
	i = 0 # number of succesive returns
	threshold = 2000
	tabuDelta = 7 
	tabuCosts = [] 
	tabuCosts.append(solutionCost)
	uniformDelta = 1
	c = 0
	flagC = True
	tour = 2.25
	tourCoef = 3
	while stop is False:
		i += 1
		c += 1
		if i % (threshold // 5) == 0:
			c = 1
			if flagC is False:
				solution = optimal
				solutionCost = optimalCost
				tabuCosts = []
				tour = 2.25
			else:
				tour = 0.125
			flagC = not flagC
		k = 1
		allNeighbors = []
		allNeighborsCosts = []
		costAllNeighbors = 0
		while k <= len(solution) // 2:
			neighbors = generateNeighbors(solution, k)
			bestNeighborCost = MAX_INT
			bestNeighborIndex = -1          
			for j in range(len(neighbors)):
				allNeighbors.append(neighbors[j])
				temp = f(neighbors[j])
				allNeighborsCosts.append(1 / temp)
				costAllNeighbors += 1 / temp
				if temp < bestNeighborCost:
					bestNeighborCost = temp
					bestNeighborIndex = j
			if bestNeighborCost < solutionCost:
				solution = neighbors[bestNeighborIndex]
				solutionCost = bestNeighborCost
				if solutionCost < optimalCost:
					optimal = solution
					optimalCost = solutionCost
					uniformDelta = 1
				break
			else: 
				k += 1
		if k > len(solution) // 2: 
			probSol = (1 / solutionCost) / ((1 / solutionCost) + costAllNeighbors)
			precision = pow(10, len(re.search('\d+\.(0*)', str(probSol)).group(1)))
			probNeighbors = neighborsProbs(allNeighborsCosts, (1 / solutionCost) + costAllNeighbors)            
			if i > threshold:
				stop = True
			else:
				newFlag = False
				probNeighbors.insert(0, probSol)
				while newFlag is False:
					if len(tabuCosts) >= len(allNeighborsCosts) * tabuDelta:
						tabuCosts = []
						break
					else:
						temp = tabuCosts[:]
						for ind in range(len(temp)):
							temp[ind] = 1 / temp[ind]
						if (list(set(allNeighborsCosts) - set(temp))) == []:
							tabuCosts = []
							ind = random.randint(len(allNeighbors)*1//3, len(allNeighbors)*2//3)
							solution = allNeighbors[ind]
							solutionCost = f(allNeighbors[ind])
							break
					probMagnitude = random.uniform(0, uniformDelta)
					sumMagnitude = 0
					for j in range(len(probNeighbors)):
						sumMagnitude += probNeighbors[j] * (precision+1 - (precision / threshold) * c)
						if probMagnitude < sumMagnitude:
							if j != 0:
								tempcost = f(allNeighbors[j-1])
								if tempcost not in tabuCosts:
									solution = allNeighbors[j-1]
									solutionCost = tempcost
									tabuCosts.append(tempcost)
									newFlag = True
									uniformDelta =  1+(precision / (threshold*tour)) * c
									break
								elif tabuDelta != 2:
									tabuDelta -= 1
	return format3Dec(optimalCost), format3Dec(initialCost), format3Dec(initialCost / optimalCost)

def runTest(numTests, description, heuristicMethod, **args):
	'''
		heuristic parameter is a function: 
		give function name as parameter: tabuSearch etc. 
		**args is arguments of heuristicMethod
	'''
	heuristicDescription(description)
	totalTime = 0.0
	for testCounter in range(numTests):
		start = time.time()
		optimalCost, initialCost, efficiency = heuristicMethod(**args)
		end = time.time()
		totalTime += end-start
		afterTestMessage(testCounter+1, optimalCost, initialCost, format8Dec(end-start), efficiency)
	afterRunMessage(format8Dec(totalTime / numTests))

numTests = 2
problemName = "1-dimensional Circle Packing"

beforeRunMessage(problemName, numTests, len(problem.problem))
runTest(numTests, "Stochastic VNS", stochasticVNS)
runTest(numTests, "SimulatedAnnealing", simulatedAnnealing)
runTest(numTests, "Tabu Search", tabuSearch)
runTest(numTests, "Iterated Local Search", iteratedLocalSearch)
runTest(numTests, "Genetic Algorithm", geneticAlgorithm, MAX_ITERATION=200)

