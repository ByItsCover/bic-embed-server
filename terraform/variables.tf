# General AWS

variable "aws_region" {
  type        = string
  description = "AWS Region"
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
  default     = "1024"
}

variable "lambda_timeout" {
  type        = number
  description = "Lambda function timout duration in seconds"
  default     = "30"
}

# Elastic Container Registry

variable "ecr_repo_name" {
  type        = string
  description = "Elastic Container Registry Repository Name"
}
