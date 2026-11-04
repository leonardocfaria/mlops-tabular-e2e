# Serves the exported production model. Build after `python -m mlops.pipeline` created serving_model/.
FROM python:3.12-slim
WORKDIR /app
COPY requirements-serve.txt .
# Optional corporate-proxy CA: docker build --secret id=ca,src=ca.crt --build-arg PIP_CERT=/ca.crt .
ARG PIP_CERT
RUN --mount=type=secret,id=ca,target=/ca.crt,required=false \
    pip install --no-cache-dir ${PIP_CERT:+--cert $PIP_CERT} -r requirements-serve.txt
COPY mlops/__init__.py mlops/data.py mlops/serve.py mlops/
COPY serving_model/ serving_model/
ENV MODEL_DIR=/app/serving_model
EXPOSE 8000
USER nobody
CMD ["uvicorn", "mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
