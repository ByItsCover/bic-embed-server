locals {
  image_uri = sensitive("${data.terraform_remote_state.bic_infra.outputs.ecr_repo_url}:latest")
}

resource "aws_lambda_function" "server_function" {
  function_name = var.lambda_name
  timeout       = 10 # seconds
  image_uri     = local.image_uri
  package_type  = "Image"

  role = aws_iam_role.api_function_role.arn
}
