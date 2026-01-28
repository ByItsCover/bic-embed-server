locals {
  ecr_repo = data.terraform_remote_state.bic_infra.outputs.embed_server_ecr_name
  lambda_role_arn = data.terraform_remote_state.bic_infra.outputs.lambda_function_role_arn
}


data "aws_ecr_image" "server_image" {
  repository_name = local.ecr_repo
  image_tag       = "latest"
}

resource "aws_lambda_function" "server_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.server_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = local.lambda_role_arn
}

resource "aws_lambda_permission" "url_public_access" {
  statement_id           = "AllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.server_function.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "public_access" {
  statement_id           = "AllowPublicAccessGenerally"
  action                 = "lambda:InvokeFunction"
  function_name          = aws_lambda_function.server_function.function_name
  principal              = "*"
}

resource "aws_lambda_function_url" "server_url" {
  function_name      = aws_lambda_function.server_function.function_name
  authorization_type = "NONE"
}
