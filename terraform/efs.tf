resource "aws_efs_file_system" "lambda" {
  encrypted = true

  tags = {
    Name = "lambda-efs"
  }
}

# Mount target in each subnet
resource "aws_efs_mount_target" "lambda_mount" {
  for_each = toset(data.aws_subnets.subnet.ids)

  file_system_id  = aws_efs_file_system.lambda.id
  subnet_id       = each.value
  security_groups = [data.terraform_remote_state.bic_infra.outputs.efs_sg_id]
}

# Access point for Lambda
resource "aws_efs_access_point" "lambda_access_point" {
  file_system_id = aws_efs_file_system.lambda.id

  root_directory {
    path = "/lambda"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  posix_user {
    gid = 1000
    uid = 1000
  }
}
