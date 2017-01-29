#!/bin/bash

python ./models/predict.py --model_path='models/tmp/model_temp.ckpt' --image_path='test_data/$1'
