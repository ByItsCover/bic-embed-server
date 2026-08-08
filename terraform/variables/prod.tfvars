aws_region  = "us-east-2"
environment = "prod"

bic_infra_workspace = "bic-infra-prod"

lambda_memory  = 3008
lambda_timeout = 30
log_level      = "DEBUG"

sqs_batch_size      = 100
sqs_max_concurrency = 100
sqs_batching_window = 5
