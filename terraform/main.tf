locals {
  lambda_role_arn    = data.terraform_remote_state.bic_infra.outputs.lambda_function_role_arn
  s3_db_uri          = data.terraform_remote_state.bic_infra.outputs.s3_db_uri
  rec_efs_access_arn = data.terraform_remote_state.bic_infra.outputs.rec_efs_access_arn
  lambda_sg_id       = data.terraform_remote_state.bic_infra.outputs.lambda_sg_id
}


resource "aws_lambda_function" "embed_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.embed_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = local.lambda_role_arn

  vpc_config {
    subnet_ids         = data.aws_subnets.subnet.ids
    security_group_ids = [local.lambda_sg_id]
  }

  file_system_config {
    arn              = local.rec_efs_access_arn
    local_mount_path = var.efs_path
  }

  logging_config {
    log_format            = "JSON"
    application_log_level = var.log_level
    system_log_level      = "INFO"
  }

  environment {
    variables = {
      ENVIRONMENT    = var.environment
      DB_URI         = local.s3_db_uri
      MODEL_ROOT_DIR = var.efs_path
    }
  }
}
