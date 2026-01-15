data "terraform_remote_state" "bic_infra" {
  backend = "remote"

  config = {
    organization = var.tfe_org_name
    workspaces = {
      name = var.bic_infra_workspace
    }
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnet_ids" "subnet" {
  vpc_id = data.aws_vpc.default.id
}
