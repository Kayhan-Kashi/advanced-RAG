FROM python:3.11-slim

WORKDIR /app

ARG PIP_INDEX_URL=https://mirror-pypi.runflare.com/simple

COPY requirements.txt .
RUN pip install -r requirements.txt --index-url ${PIP_INDEX_URL}

RUN pip install -vvv torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install -vvv transformers==4.44.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv sentence-transformers==3.2.0 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv langchain-huggingface==1.2.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv einops==0.8.2 --index-url ${PIP_INDEX_URL}
RUN pip install -vvv faiss-cpu==1.14.2 --index-url ${PIP_INDEX_URL}
RUN pip install huggingface-hub --index-url ${PIP_INDEX_URL}
RUN pip install langchain --index-url ${PIP_INDEX_URL}
RUN pip install  langchain-core --index-url ${PIP_INDEX_URL}
RUN pip install langchain-community --index-url ${PIP_INDEX_URL}
RUN pip install rank_bm25 langchain-community scipy --index-url ${PIP_INDEX_URL}
RUN pip install FlagEmbedding --index-url ${PIP_INDEX_URL}


COPY shared/ ./shared/
RUN pip install ./shared --index-url ${PIP_INDEX_URL}

COPY services/llm-service/ ./

ENV PYTHONPATH=/app

CMD ["python", "-m", "src.workers.worker"]