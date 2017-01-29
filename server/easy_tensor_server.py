from flask import Flask, render_template, request
import flowchart_parser
import json

app = Flask(__name__)
parser = flowchart_parser.FlowchartParser()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_model():
    json_received = request.form['modelDescription']
    # json_received = request.data
    print('JSON:')
    print(json_received)
    parser.parse_and_output(json_received)
    return 'test output'


if __name__ == '__main__':
    test_json = r'{ "class": "go.GraphLinksModel", "nodeDataArray": [{"text":"Perceptron Layer", "figure":"RoundedRectangle", "fill":"lightyellow", "para":"Dimension:2\nActivation: ReLU\nmulti:3", "key":-3, "loc":"200 250"}, {"text":"Input", "figure":"Circle", "fill":"lightgray", "para":"Dimension:5\nlearning rate:0.3", "key":-1, "loc":"190 90"}, {"text":"Output", "figure":"Circle", "fill":"lightgray", "para":"Dimension:4", "key":-2, "loc":"180 460"}], "linkDataArray": [{"from":-1, "to":-3, "points":[190,156.40299940829067,190,166.40299940829067,190,177.68972856443162,200,177.68972856443162,200,188.9764577205726,200,198.9764577205726]}, {"from":-3, "to":-2, "points":[200,301.02354227942743,200,311.02354227942743,200,347.3102714355684,180,347.3102714355684,180,383.59700059170933,180,393.59700059170933]}]}'
    app.run()
    # parser.parse_and_output(test_json)
