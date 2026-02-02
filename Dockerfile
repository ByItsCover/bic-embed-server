# Build Stage

ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim AS build

WORKDIR /build_dir
COPY download_model.py requirements_build.txt ./

RUN pip install --no-cache-dir -r requirements_build.txt
RUN python download_model.py ./

# Deploy Stage

FROM public.ecr.aws/lambda/python:${PYTHON_VERSION} AS deploy

WORKDIR ${LAMBDA_TASK_ROOT}
COPY --from=build /build_dir/clip_model/clip_quantized.onnx ./clip_model/
COPY server.py helpers.py requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "server.handler" ]
