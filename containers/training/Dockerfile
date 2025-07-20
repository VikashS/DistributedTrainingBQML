# Start from base image
FROM base as training

# Install additional training dependencies
RUN pip install scikit-learn==1.2.2 xgboost==1.7.5 lightgbm==3.3.5

# Copy training-specific files
COPY containers/training/entrypoint.sh /app/entrypoint.sh
COPY containers/training/model_configs/ /app/configs/models/
COPY configs/models/ /app/configs/models/

# Set permissions
RUN chmod +x /app/entrypoint.sh

# Set environment variables
ENV MODEL_CONFIG_DIR=/app/configs/models

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]