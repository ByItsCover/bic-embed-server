resource "aws_api_gateway_rest_api" "embed_api" {
  name        = "embed-api"
  description = "Embed API Gateway"
}

resource "aws_api_gateway_resource" "items" {
  rest_api_id = aws_api_gateway_rest_api.embed_api.id
  parent_id   = aws_api_gateway_rest_api.embed_api.root_resource_id
  path_part   = "items"
}

resource "aws_api_gateway_method" "post" {
  rest_api_id   = aws_api_gateway_rest_api.embed_api.id
  resource_id   = aws_api_gateway_resource.items.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_post" {
  rest_api_id = aws_api_gateway_rest_api.embed_api.id
  resource_id = aws_api_gateway_resource.items.id
  http_method = aws_api_gateway_method.post.http_method

  integration_http_method = "POST"
  type                   = "AWS_PROXY"
  uri                    = aws_lambda_function.embed_function.invoke_arn
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.embed_function.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.embed_api.execution_arn}/*/*"
}

resource "aws_api_gateway_deployment" "embed_deployment" {
  depends_on = [
    aws_api_gateway_integration.lambda_post
  ]

  rest_api_id = aws_api_gateway_rest_api.embed_api.id
}

resource "aws_api_gateway_stage" "embed_stage" {
  deployment_id = aws_api_gateway_deployment.embed_deployment.id
  rest_api_id   = aws_api_gateway_rest_api.embed_api.id
  stage_name    = var.environment
}
