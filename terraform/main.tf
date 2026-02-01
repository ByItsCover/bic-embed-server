locals {
  ecr_repo = data.terraform_remote_state.bic_infra.outputs.embed_server_ecr_name
  lambda_role_arn = data.terraform_remote_state.bic_infra.outputs.lambda_function_role_arn
  api_gw_id = data.terraform_remote_state.bic_infra.outputs.api_gw_id
  api_gw_arn = data.terraform_remote_state.bic_infra.outputs.api_gw_arn
}


# Lambda Function

data "aws_ecr_image" "embed_image" {
  repository_name = local.ecr_repo
  image_tag       = "latest"
}

resource "aws_lambda_function" "embed_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.embed_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = local.lambda_role_arn
}

resource "aws_lambda_permission" "public_access" {
  statement_id           = "AllowExecutionFromAPIGateway"
  action                 = "lambda:InvokeFunction"
  function_name          = aws_lambda_function.embed_function.function_name
  principal              = "apigateway.amazonaws.com"

  source_arn = "${local.api_gw_arn}/*/*"
}

# API Gateway

resource "aws_apigatewayv2_integration" "lambda_post" {
  api_id           = local.api_gw_id
  
  integration_type = "AWS_PROXY"
  integration_method = "POST"
  integration_uri    = aws_lambda_function.embed_function.invoke_arn

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_apigatewayv2_route" "predict_post" {
  api_id    = local.api_gw_id

  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_post.id}"
  authorization_type = "AWS_IAM"
}

/*
resource "aws_apigatewayv2_route" "default_get" {
  api_id    = local.api_gw_id

  route_key = "GET /"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_get.id}"
  authorization_type = "AWS_IAM"
}
*/

resource "aws_apigatewayv2_stage" "embed_stage" {
  api_id      = local.api_gw_id

  name        = var.environment
  auto_deploy = true
}
