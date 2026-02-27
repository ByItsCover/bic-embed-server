# General AWS

variable "aws_region" {
  type        = string
  description = "AWS Region"
}

variable "environment" {
  type        = string
  description = "Deployment Environment"
}

# Terraform Cloud

variable "tfe_org_name" {
  type        = string
  description = "Terraform Cloud organization name"
  default     = "ByItsCover"
}

variable "bic_infra_workspace" {
  type        = string
  description = "Terraform Cloud Workspace BIC-Infra name"
}

# Lambda

variable "lambda_name" {
  type        = string
  description = "Name of Lambda Function"
  default     = "embed-server-lambda"
}

variable "lambda_memory" {
  type        = number
  description = "Memory in MB alloted to Lambda function"
  default     = 1024
}

variable "lambda_timeout" {
  type        = number
  description = "Lambda function timeout duration in seconds"
  default     = 30
}

# SQS

variable "sqs_batch_size" {
  type        = number
  description = "Max size of SQS batch to lambda"
  default     = 10
}

variable "sqs_max_concurrency" {
  type        = number
  description = "Maximum concurrency of batches for SQS with lambda"
  default     = 100
}

variable "sqs_batching_window" {
  type        = number
  description = "Maximum batching window for SQS with lambda"
  default     = 0
}
