#!/bin/bash

python -m fraud.prediction.main \
  --model-path=$MODEL_PATH \
  --input-data=$INPUT_DATA \
  --predictions-output=$PREDICTIONS_OUTPUT \
  --batch-size=${BATCH_SIZE:-1000}