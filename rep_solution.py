import random
import problem

def generateRandomSolution():
	# Shuffle the problem and start with random solution
	return random.sample(problem.problem, len(problem.problem))

def initialSolution():
	# Shuffle the problem and start with random solution
	return random.sample(problem.problem, len(problem.problem))