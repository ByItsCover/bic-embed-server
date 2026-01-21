ARG PYTHON_VERSION=3.13
FROM public.ecr.aws/lambda/python:$PYTHON_VERSION

WORKDIR /app
COPY . ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN python ${LAMBDA_TASK_ROOT}/download_model.py ${LAMBDA_TASK_ROOT}

EXPOSE 8000

CMD [ "server.handler" ]
