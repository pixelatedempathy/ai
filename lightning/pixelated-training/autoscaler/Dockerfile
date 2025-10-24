# Autoscaler for Wayfarer Model Instances
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install \
    docker \
    prometheus-client \
    requests \
    psutil \
    pyyaml

# Copy autoscaler script
COPY autoscaler.py .
COPY config.yml .

# Create non-root user
RUN useradd -m autoscaler && chown -R autoscaler:autoscaler /app
USER autoscaler

CMD ["python", "autoscaler.py"]