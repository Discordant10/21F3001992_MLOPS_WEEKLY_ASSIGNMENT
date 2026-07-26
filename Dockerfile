FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Create Feast registry during image build
RUN mkdir -p /app/feature_repo/data && \
    cd /app/feature_repo && \
    feast apply

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]