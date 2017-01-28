import jinja2


class FlowchartParser:
    def __init__(self):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader('../models')
        )


    def parse_and_output(self, josn_str, filename):
        template = self.env.get_template('model_template.py')
        print(template.render())
