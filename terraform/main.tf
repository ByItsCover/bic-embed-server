data "aws_ecr_image" "server_image" {
  repository_name = var.ecr_repo_name
  image_tag       = "latest"
}

resource "aws_lambda_function" "server_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.server_image.image_uri
  package_type  = "Image"

  memory_size = 512
  timeout     = 30 # seconds

  role = aws_iam_role.api_function_role.arn
}
