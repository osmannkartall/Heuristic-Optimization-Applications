LEFT_ARROW = '←'
RIGHT_ARROW = '→'

def createIntegerArray(n):
    array = list()
    for i in range(1, n+1):
        array.append(i);
    return array

def createDirectionArray(n):
    array = list()
    for i in range(n):
        array.append(LEFT_ARROW);
    return array    

def findLargestMobile(numbers, directions):
    largest = 0
    length = len(numbers)
    for i in range(0, length):
        # 1 < m < n and left arrow
        if directions[i] == LEFT_ARROW and i > 0 and numbers[i] > numbers[i-1] and numbers[i] > largest:
            largest = numbers[i]
        # 1 < m < n and right arrow
        elif i < length-1 and directions[i] == RIGHT_ARROW and numbers[i] > numbers[i+1] and numbers[i] > largest:
            largest = numbers[i]
    if largest == 0:
        i = -1
    else:
        i = numbers.index(largest)
    return largest, i;

def switchDirections(numbers, directions, m):
    for i in range(len(numbers)):
        if numbers[i] > m:
            if directions[i] == LEFT_ARROW:
                directions[i] = RIGHT_ARROW
            else:
                directions[i] = LEFT_ARROW
    return 0

def switchAdjacents(numbers, directions, m, m_index, a_index):
    numbers[m_index] = numbers[a_index]
    numbers[a_index] = m
    temp = directions[m_index] 
    directions[m_index] = directions[a_index]
    directions[a_index] = temp

def generatePermutation(numbers, directions, permutations):
    m, m_index = findLargestMobile(numbers, directions)
    while m != 0:
        m, m_index = findLargestMobile(numbers, directions)
        # Switch m and the adjacent integer its arrow points to
        # Also, switch their direction values
        if directions[m_index] == LEFT_ARROW:
            switchAdjacents(numbers, directions, m, m_index, m_index-1)
        elif m_index != len(numbers)-1:
            switchAdjacents(numbers, directions, m, m_index, m_index+1)
        switchDirections(numbers, directions, m)
        # To prevent shallow copy use [:]
        permutations.append(numbers[:])
    # TODO: This is not good.
    permutations.pop()


def generator(n):
    numbers = createIntegerArray(n)
    directions = createDirectionArray(n)
    # To prevent shallow copy use [:]   
    permutations = list()
    permutations.append(numbers[:])

    if n < 0:
        print('Choose a non-negative integer!')
        return -1
    elif n == 0 or n == 1:
        return 1
    else:
        generatePermutation(numbers, directions, permutations)
        return permutations
        # return 'fact(' + str(n) + ') = ' + str(1 + generatePermutation(numbers,directions))

# Run the generator program
# generator(4)