locals {
  ecr_repo        = data.terraform_remote_state.bic_infra.outputs.embed_server_ecr_name
  lambda_role_arn = data.terraform_remote_state.bic_infra.outputs.lambda_function_role_arn
  api_gw_arn      = data.terraform_remote_state.bic_infra.outputs.api_gw_arn
  s3_db_uri          = data.terraform_remote_state.bic_infra.outputs.s3_db_uri
}


resource "aws_lambda_function" "embed_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.embed_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = local.lambda_role_arn

  snap_start {
    apply_on = "PublishedVersions"
  }

  environment {
    variables = {
      ENVIRONMENT = var.environment
      DB_URI = local.s3_db_uri
    }
  }
}

resource "aws_lambda_permission" "public_access" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.embed_function.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${local.api_gw_arn}/*/*"
}
