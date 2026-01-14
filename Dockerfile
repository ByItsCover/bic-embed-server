ARG PYTHON_VERSION=3.13
#FROM python:$PYTHON_VERSION-slim
FROM public.ecr.aws/lambda/python:$PYTHON_VERSION
#COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter

WORKDIR /app
COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN python ${LAMBDA_TASK_ROOT}/download_model.py ${LAMBDA_TASK_ROOT}

# ENV AWS_LWA_ASYNC_INIT=true
# ENV PORT=8000
EXPOSE 8000

#CMD ["fastapi", "run", "/app/server.py", "--port", "8000"]
CMD [ "server.handler" ]
