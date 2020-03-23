import random as rand
import numpy as item


class Perceptron:

    def __init__(self, length, weights=[]):
        self.weights = weights
        self.length = length
        self.learning_rate = 0.1

        '''
        for _ in range(length):
            number = rand.randint(-1, 1)
            weights.append(number)

    
        def guess(self, inputs=[]):
        sum = 0
        length = len(self.weights)
        for i in range(length):

            sum += inputs[i] * self.weights[i]

        output = item.sign(sum)

        return output
    '''

    def hardlim(self, val):
        return 0 if val < 0 else 1

    def train(self, inp, target):

        N,n = inp.shape

        w1 = item.random.randn(n, 1)
        w2 = item.random.randn(n, 1)

        E=1
        while E != 0:

            E = 0

            for i in range(N):
                y1 = self.hardlim(item.dot(inp[i], w1))
                y2 = self.hardlim(item.dot(inp[i], w2))

                e1 = target[i][0] - y1
                e2 = target[i][1] - y2

                w1 += (self.learning_rate * e1 * inp[i].reshape(n, 1)[0])
                w2 += (self.learning_rate * e2 * inp[i].reshape(n, 1)[0])

                E += e1**2
                E += e2**2

                print(e1, e2)

'''
class Character:

    def __init__(self, twoDArray=[], label=''):
        self.twoDArray = twoDArray
        self.label = label
'''

if __name__ == "__main__":

    perc = Perceptron(3)

    data = item.array([
        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1],  # H
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # I
    ])

    labels = item.array([
        [1, 0],
        [0, 1],
    ])

    '''
    characters = []

    H = Character(item.array([1, 1, 0, 1, 1, 1, 1, 1, 0, 1]), "H")
    I = Character(item.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0]), "I")

    characters= []

    characters.append(H)
    characters.append(I)
    
    '''
    perc.train(data, labels)