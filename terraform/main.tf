locals {
  ecr_repo        = data.terraform_remote_state.bic_infra.outputs.embed_server_ecr_name
  lambda_role_arn = data.terraform_remote_state.bic_infra.outputs.lambda_function_role_arn
  s3_db_uri       = data.terraform_remote_state.bic_infra.outputs.s3_db_uri
}


resource "aws_lambda_function" "embed_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.embed_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = local.lambda_role_arn

  logging_config {
    log_format            = "TEXT"
    application_log_level = var.log_level
    system_log_level      = "INFO"
  }

  environment {
    variables = {
      ENVIRONMENT = var.environment
      DB_URI      = local.s3_db_uri
    }
  }
}
