data "aws_ecr_image" "server_image" {
  repository_name = var.ecr_repo_name
  image_tag       = "latest"
}

resource "aws_lambda_function" "server_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.server_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  role = aws_iam_role.api_function_role.arn
}

resource "aws_lambda_function_url" "server_url" {
  function_name      = aws_lambda_function.server_function.function_name
  authorization_type = "AWS_IAM"
}

resource "aws_lambda_permission" "url_public_access" {
  statement_id           = "AllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.server_function.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}
