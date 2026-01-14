ARG PYTHON_VERSION=3.13
FROM python:$PYTHON_VERSION-slim
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python download_model.py

ENV AWS_LWA_ASYNC_INIT=true
ENV PORT=8000
EXPOSE 8000

CMD ["fastapi", "run", "/app/server.py", "--port", "8000"]
