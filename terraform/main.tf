data "aws_ecr_image" "server_image" {
  repository_name = var.ecr_repo_name
  image_tag       = "latest"
}

# The Bootstrap Lambda
resource "aws_lambda_function" "bootstrap_function" {
  function_name = "efs-bootstrap"
  image_uri     = data.aws_ecr_image.server_image.image_uri
  package_type  = "Image"

  # OVERRIDE the command to run the bootstrap script instead
  image_config {
    command = ["bootstrap.handler"]
  }

  # Boost this one's timeout since it's downloading a big model
  timeout = 120 # 2 minutes

  vpc_config {
    subnet_ids         = data.aws_subnet_ids.subnet.ids
    security_group_ids = [data.terraform_remote_state.bic_infra.outputs.efs_sg_id]
  }

  file_system_config {
    arn              = aws_efs_access_point.lambda_access_point.arn
    local_mount_path = "/mnt/data"
  }

  # Ensure EFS is ready before Lambda creation
  depends_on = [aws_efs_mount_target.lambda_mount]

  role = aws_iam_role.api_function_role.arn
}

resource "aws_lambda_function" "server_function" {
  function_name = var.lambda_name
  image_uri     = data.aws_ecr_image.server_image.image_uri
  package_type  = "Image"

  memory_size = var.lambda_memory
  timeout     = var.lambda_timeout

  vpc_config {
    subnet_ids         = data.aws_subnet_ids.subnet.ids
    security_group_ids = [data.terraform_remote_state.bic_infra.outputs.efs_sg_id]
  }

  file_system_config {
    arn              = aws_efs_access_point.lambda_access_point.arn
    local_mount_path = "/mnt/data"
  }

  # Ensure EFS is ready before Lambda creation
  depends_on = [aws_efs_mount_target.lambda_mount]

  role = aws_iam_role.api_function_role.arn
}

resource "aws_lambda_permission" "url_public_access" {
  statement_id           = "AllowPublicAccess"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.server_function.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

resource "aws_lambda_permission" "public_access_bootstrap" {
  statement_id           = "AllowPublicAccessGenerally"
  action                 = "lambda:InvokeFunction"
  function_name          = aws_lambda_function.bootstrap_function.function_name
  principal              = "*"
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
