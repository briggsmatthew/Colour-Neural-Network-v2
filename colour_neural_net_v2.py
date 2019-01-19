import numpy as np


class ColourNetwork(object):
    """
    A pre-structured neural network for identifying what an RGB input is and
    returning a colour string (along with the confidence for each other colour).
    Usage:
    a = ColourNetwork()
    a.train_from_file("training1.txt")
    a.interact([255, 0, 0]) #See what (255, 0, 0) is -> Red
    """

    def __init__(self):
        self.training_in = None
        self.training_out = None
        self.l0 = None
        self.l1 = None

        np.random.seed(100)
        self.w0 = 2 * np.random.random((3, 9)) - 1

        self.colours = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo",
                        "Violet", "Black", "White"]

    def nonlinear(self, x, deriv=False):
        """Sigmoid function"""
        if deriv:
            return x * (1 - x)

        return 1 / (1 + np.exp(-x))

    def train_network(self):
        for i in range(10000):
            self.l0 = self.training_in
            self.l1 = self.nonlinear(np.dot(self.l0, self.w0))

            l1_error = self.training_out - self.l1

            l1_delta = l1_error * self.nonlinear(self.l1, True)

            self.w0 += np.dot(self.l0.T, l1_delta)

    def train_from_file(self, filepath):
        with open(filepath) as f:
            f = f.read()

        ins = eval(f.split("---")[0])

        outs = f.split("---")[1].split("\n")
        outs_list = []

        for line in outs:
            line = line.split(" | ")
            if len(line) == 2:
                if "," in line[1]:
                    line[1] = line[1][:-1]
                num = int(line[1])

                for i in range(num):
                    outs_list.append(eval(line[0]))

        self.training_in = np.array(ins)
        self.training_out = np.array(outs_list)

        for i in self.training_in:
            for x in i:
                x -= 0.5

        for i in self.training_out:
            for x in i:
                x -= 0.5

        self.train_network()

    def max_output(self, outputs):
        max_out = 0
        index = 0
        for i in range(len(outputs)):
            if outputs[i] > max_out:
                #print(outputs[i], max_out)
                max_out = outputs[i]
                index = i

        return index

    def interact(self, in_):
        # for i in in_:
            #i -= 0.5
        l0 = in_
        l1 = self.nonlinear(np.dot(l0, self.w0))

        max_ = self.max_output(l1)

        for i in range(len(l1)):
            print(self.colours[i], ": ", round(l1[i], 6) * 100, "%", sep="")

        for i in range(len(l1)):
            if max_ == i:
                print("Best idea: " + self.colours[i])

    def weight_correlation(self):
        print("Red input neuron correlates as such:")
        for i in range(len(self.w0[0])):
            print(self.colours[i], ": ", self.w0[0][i])

        print("Green input neuron correlates as such:")
        for i in range(len(self.w0[1])):
            print(self.colours[i], ": ", self.w0[1][i])

        print("Blue input neuron correlates as such:")
        for i in range(len(self.w0[2])):
            print(self.colours[i], ": ", self.w0[2][i])
