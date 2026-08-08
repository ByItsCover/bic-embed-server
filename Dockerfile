# Build Stage

ARG PYTHON_VERSION=3.13
ARG FUNCTION_DIR="/app/function/"

FROM python:${PYTHON_VERSION}-slim AS build

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    unzip \
    libcurl4-openssl-dev

ARG FUNCTION_DIR

RUN mkdir -p ${FUNCTION_DIR}

#COPY download_model.py build_requirements.txt requirements.txt ./
COPY requirements.txt ./

#RUN pip install --no-cache-dir -r build_requirements.txt
#RUN python download_model.py ${FUNCTION_DIR}
RUN pip install --no-cache-dir awslambdaric --target ${FUNCTION_DIR}
RUN pip install --no-cache-dir -r requirements.txt --target ${FUNCTION_DIR}

# Deploy Stage

FROM gcr.io/distroless/python3-debian13 AS deploy

ARG FUNCTION_DIR

WORKDIR ${FUNCTION_DIR}
ENV ROOT_DIR=${FUNCTION_DIR}

COPY --from=build ${FUNCTION_DIR} ${FUNCTION_DIR}
COPY ./src ./

ENTRYPOINT [ "python3", "-m", "awslambdaric"]

CMD [ "main.lambda_handler" ]
