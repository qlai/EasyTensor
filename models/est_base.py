class EstBase(object):
    def __init__(self, input_dim, output_dim, costfunc, learning_rate, optimizer):
        '''
        :param input_dim:
        :param output_dim:
        :param costfunc: function
        :param learning_rate: float
        :param optimizer: str, to specify the optimizer to use. candidate: GD, ADAM
        :return:
        '''
        optimizer = optimizer.upper()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.costfunc = costfunc
        self.learning_rate = learning_rate
        self.optimizer = optimizer


