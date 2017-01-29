###HackCambridge 2017 Project 
by Dao Zhou, Minxuan Xie, Renqiao Zhang, Qiuying Lai
![EasyTensor](https://github.com/qlai/EasyTensor/blob/master/easytensor.png)
##EasyTensor: GUI for Simple Models in Tensorflow
Helping beginners and students to understand the structure of Tensorflow better by providing a GUI where standard neural net models written in Tensorflow (Python) can be generated.

This uses MNIST Data as an example. `debug*.py` shows how the models can be trained and used.

Dependencies:
`tensorflow, jinja2, flask, opencv`

Examples:
- To draw a model use the following on terminal:
    `cd server`
    `python easy_tensor_server.py` 
  to launch drawer on localhost
- Click `generate` on the webpage to generate model file out local directory, this will output a file in 'models' directory in the format `[model_type]_output.py`
- Follow template on `debug0` for a MultiLayerPerceptron Model, `debug1` to run a CNN Model

Further Instructions:
- You must use a perceptron model after layers of CNN for flattening before output layer
