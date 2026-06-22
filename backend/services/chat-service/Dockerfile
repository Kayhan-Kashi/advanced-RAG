FROM python:3.11-slim

WORKDIR /app

ARG PIP_INDEX_URL=https://mirror-pypi.runflare.com/simple

COPY requirements.txt .
RUN pip install -r requirements.txt --index-url ${PIP_INDEX_URL}


RUN pip install python-multipart --index-url ${PIP_INDEX_URL}

COPY backend/shared/ ./shared/
RUN pip install ./shared --index-url ${PIP_INDEX_URL}

COPY backend/services/chat-service/src/ ./

ENV PYTHONPATH=/app
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]