# Start from base image
FROM base as prediction

# Copy prediction-specific files
COPY containers/prediction/entrypoint.sh /app/entrypoint.sh

# Set permissions
RUN chmod +x /app/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]