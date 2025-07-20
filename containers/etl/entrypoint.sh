#!/bin/bash

python -m fraud.etl.main \
  --input-path=$INPUT_PATH \
  --output-path=$OUTPUT_PATH \
  --config-path=/app/configs/etl_config.yaml