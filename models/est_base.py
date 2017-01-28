class EstBase(object):
    def __init__(self, input_dim, output_dim, costfunc, learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.costfunc = costfunc
        self.learning_rate = learning_rate


