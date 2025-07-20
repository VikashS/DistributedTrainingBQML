#!/bin/bash

python -m fraud.training.main \
  --input-data=$INPUT_DATA \
  --model-output=$MODEL_OUTPUT \
  --model-type=$MODEL_TYPE \
  --config-path=/app/configs/models/${MODEL_TYPE}.yaml