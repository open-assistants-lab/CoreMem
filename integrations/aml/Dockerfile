# CoreMem AML adapter — built from the submitted repository so the
# evaluated code is exactly what the platform reproduces.
FROM python:3.11-slim

WORKDIR /app

# Build CoreMem from this repository (the academic submission itself)
COPY . /app
RUN pip install --no-cache-dir . \
 && pip install --no-cache-dir -r integrations/aml/requirements.txt

# The AML adapter entrypoint (copied to the app root for a flat import path)
COPY integrations/aml/server.py /app/aml_server.py

ENV COREMEM_PATH=/data/memory
ENV AML_STRATEGY=episodic
ENV AML_TOP_K=100
ENV AML_MIN_SCORE=0.0

EXPOSE 8000
CMD ["uvicorn", "aml_server:app", "--host", "0.0.0.0", "--port", "8000"]
