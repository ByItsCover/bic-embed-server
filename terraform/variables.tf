# Terraform Cloud

variable "tfe_org_name" {
  type        = string
  description = "Terraform Cloud organization name"
  default     = "ByItsCover"
}

variable "bic_infra_workspace" {
  type = string
  description = "Terraform Cloud Workspace BIC-Infra name"
}

# Lambda

variable "lambda_name" {
  type = string
  description = "describe your variable"
  default = "embed-server-lambda"
}
