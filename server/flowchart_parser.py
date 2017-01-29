import jinja2
import json

class ModelData(object):
    def __init__(self, input_dim, output_dim, activations, learning_rate, optimizer='\'ADAM\''):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activations = activations
        self.learning_rate = learning_rate
        self.optimizer = optimizer

class MultilayerPerceptronData(ModelData):
    def __init__(self, input_dim, output_dim, hidden_layers_dims, activations, \
                     learning_rate, optimizer = '\'GD\'', dropout = 'False', costfunc = 'utils.cross_entropy'):
        super(MultilayerPerceptronData, self).__init__(input_dim, output_dim, activations, learning_rate, optimizer)
        self.hidden_layers_dims = hidden_layers_dims
        self.dropout = dropout
        self.costfunc = costfunc

class ConvolutionNNData(ModelData):
    def __init__(self, input_dim, output_dim, hidden_patches, hidden_layers_dims, \
                 activations, learning_rate, optimizer='\'GD\'', dropout = False, \
                 costfunc = 'utils.cross_entropy'):
        super(ConvolutionNNData, self).__init__(input_dim, output_dim, activations, learning_rate, optimizer)
        self.hidden_patches = hidden_patches
        self.hidden_layers_dims = hidden_layers_dims
        self.dropout = dropout
        self.costfunc = costfunc

class FlowchartParser:
    def __init__(self):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('../models')
        )

    def _try_parse_int(self, s):
        ret = None
        try:
            ret = int(s)

        except ValueError:
            pass
        # ret = int(s)

        return ret

    def _parse_para_string(self, para_string):
        para_list = para_string.split('\n')
        paras = [para.split(':')[1].lstrip() for para in para_list]
        return paras

    def _parse_input_node(self, input_node):
        input_para = input_node['para']
        paras = self._parse_para_string(input_para)
        input_dim_2d = [int(i) for i in paras[0].split('*')]
        learning_rate = float(paras[1])
        optimizer = '\'' + paras[2] + '\''
        return input_dim_2d, learning_rate, optimizer

    def _parse_nodes(self, head_key, key_to_node, key_to_next, model_type):
        node = key_to_node[head_key]
        input_dim_2d, learning_rate, optimizer = self._parse_input_node(node)
        model_data = None
        layer_dims = []
        activations = []
        output_dim = None
        patch_sizes = []
        key_num = len(key_to_node)

        node = key_to_next[head_key]

        while node['key'] in key_to_node:
            if node['text'].lower() == 'output':
                paras = self._parse_para_string(node['para'])
                output_dim = self._try_parse_int(paras[0])
                break

            if model_type == 0:
                # Multi-layer perceptron model
                para_string = node['para']
                paras = self._parse_para_string(para_string)
                layer_dim = int(paras[0])
                activation = paras[1].lower()
                multi_layer_num = self._try_parse_int(paras[2])
                if multi_layer_num:
                    layer_dims.extend([layer_dim] * multi_layer_num)
                    activations.extend([activation] * multi_layer_num)
                else:
                    layer_dims.append(layer_dim)
                    activations.append(activation)

            if model_type == 1:
                # Convolution neural network model
                para_string = node['para']
                paras = self._parse_para_string(para_string)
                patch_size_tuple = paras[0]
                layer_name = node['text'].lower()
                if layer_name == 'perceptron layer':
                    patch_size = None
                    layer_dim = int(paras[0])
                    activation = paras[1].lower()
                    multi_layer_num = self._try_parse_int(paras[2])
                    if multi_layer_num:
                        layer_dims.extend([layer_dim] * multi_layer_num)
                        activations.extend([activation] * multi_layer_num)
                        patch_sizes.extend([patch_size for _ in range(multi_layer_num)])
                    else:
                        layer_dims.append(layer_dim)
                        activations.append(activation)
                        patch_sizes.append(patch_size)
                else:
                    patch_size = [int(e) for e in patch_size_tuple.split('*')]
                    layer_dim = int(paras[1])
                    activation = paras[2].lower()
                    multi_layer_num = self._try_parse_int(paras[3])
                    if multi_layer_num:
                        layer_dims.extend([layer_dim] * multi_layer_num)
                        activations.extend([activation] * multi_layer_num)
                        patch_sizes.extend([patch_size[:] for _ in range(multi_layer_num)])
                    else:
                        layer_dims.append(layer_dim)
                        activations.append(activation)
                        patch_sizes.append(patch_size)

            node = key_to_next[node['key']]

        if model_type == 0:
            model_data = MultilayerPerceptronData(input_dim_2d[0], output_dim, layer_dims, activations, \
                    learning_rate, optimizer)
        elif model_type == 1:
            model_data = ConvolutionNNData(input_dim_2d, output_dim, patch_sizes, layer_dims, \
                                           activations, learning_rate, optimizer)

        elif model_type == 2:
            pass

        return model_data

    def _parse_graph_description(self, graph_description_dict):
        nodes = graph_description_dict['nodeDataArray']
        key_to_node = {}
        head_key = None
        for node in nodes:
            key = node['key']
            key_to_node[key] = node
            if node['text'].lower() == 'input':
                head_key = key

        key_to_next = {}
        for link in graph_description_dict['linkDataArray']:
            from_key = link['from']
            to_key = link['to']
            key_to_next[from_key] = key_to_node[to_key]

        # Parse input dimension
        input_node = key_to_node[head_key]

        # Find the type of model
        first_layer_node = key_to_next[head_key]
        model_type = None
        first_layer_name = first_layer_node['text']
        if first_layer_name.lower() == 'perceptron layer':
            model_type = 0

        elif first_layer_name.lower() == 'convolution layer':
            model_type = 1

        elif first_layer_name.lower() == 'recurrent layer':
            model_type = 2

        model_data = self._parse_nodes(head_key, key_to_node, key_to_next, model_type)

        return model_data


    def _parse_multilayer_perceptron_and_output(self, model_data, filename):
        # input_dim = '8'
        # output_dim = '2'
        # hidden_layers_dims_list = [4, 5, 6, 7, 8]
        # activations = "['relu', 'relu', 'softplus', 'sigmoid', 'tanh']"
        # learning_rate = '0.563'
        # dropout = 'True'

        template_filename = 'multilayer_perceptron_template.py'
        template = self.env.get_template(template_filename)
        # hidden_layers_dims_json_str = json.dumps(model_data.hidden_layers_dims)
        hidden_layers_dims = model_data.hidden_layers_dims
        layer_num = len(model_data.hidden_layers_dims)
        render_output = template.render(input_dim=model_data.input_dim, output_dim=model_data.output_dim,
                                        hidden_layers_dims=hidden_layers_dims, layer_num=layer_num, \
                                        activations=model_data.activations, learning_rate=model_data.learning_rate, \
                                        dropout=model_data.dropout, costfunc=model_data.costfunc, \
                                        optimizer=model_data.optimizer)
        with open('../models/' + filename, 'w') as f:
            f.write(render_output)

    def _parse_convolution_nn_and_output(self, model_data, filename):
        template_filename = 'convolution_nn_template.py'
        template = self.env.get_template(template_filename)
        hidden_layers_dims = model_data.hidden_layers_dims
        layer_num = len(model_data.hidden_layers_dims)
        render_output = template.render(input_dim=model_data.input_dim, output_dim=model_data.output_dim,
                                        hidden_layers_dims=hidden_layers_dims, layer_num=layer_num, \
                                        activations=model_data.activations, learning_rate=model_data.learning_rate, \
                                        dropout=model_data.dropout, costfunc=model_data.costfunc, \
                                        optimizer=model_data.optimizer, hidden_patches=model_data.hidden_patches)
        with open('../models/' + filename, 'w') as f:
            f.write(render_output)

    def parse_and_output(self, json_str):
        graph_description_dict = json.loads(json_str)
        model_data = self._parse_graph_description(graph_description_dict)
        if not model_data:
            print('Error in parsing')
        else:
            if isinstance(model_data, MultilayerPerceptronData):
                output_filename = 'multilayer_perceptron_output.py'
                self._parse_multilayer_perceptron_and_output(model_data, output_filename)
            elif isinstance(model_data, ConvolutionNNData):
                output_filename = 'convolution_nn_output.py'
                self._parse_convolution_nn_and_output(model_data, output_filename)
