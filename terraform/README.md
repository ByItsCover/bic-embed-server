<!-- BEGIN_TF_DOCS -->
## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.2 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | ~> 5.92 |

## Providers

| Name | Version |
|------|---------|
| <a name="provider_aws"></a> [aws](#provider\_aws) | 5.100.0 |
| <a name="provider_terraform"></a> [terraform](#provider\_terraform) | n/a |

## Modules

No modules.

## Resources

| Name | Type |
|------|------|
| [aws_lambda_event_source_mapping.sqs_trigger](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lambda_event_source_mapping) | resource |
| [aws_lambda_function.embed_function](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/lambda_function) | resource |
| [aws_ecr_image.embed_image](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/data-sources/ecr_image) | data source |
| [terraform_remote_state.bic_infra](https://registry.terraform.io/providers/hashicorp/terraform/latest/docs/data-sources/remote_state) | data source |

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_aws_region"></a> [aws\_region](#input\_aws\_region) | AWS Region | `string` | n/a | yes |
| <a name="input_bic_infra_workspace"></a> [bic\_infra\_workspace](#input\_bic\_infra\_workspace) | Terraform Cloud Workspace BIC-Infra name | `string` | n/a | yes |
| <a name="input_environment"></a> [environment](#input\_environment) | Deployment Environment | `string` | n/a | yes |
| <a name="input_lambda_memory"></a> [lambda\_memory](#input\_lambda\_memory) | Memory in MB alloted to Lambda function | `number` | `1024` | no |
| <a name="input_lambda_name"></a> [lambda\_name](#input\_lambda\_name) | Name of Lambda Function | `string` | `"embed-server-lambda"` | no |
| <a name="input_lambda_timeout"></a> [lambda\_timeout](#input\_lambda\_timeout) | Lambda function timeout duration in seconds | `number` | `30` | no |
| <a name="input_log_level"></a> [log\_level](#input\_log\_level) | Application log level for Lambda Function | `string` | `"DEBUG"` | no |
| <a name="input_sqs_batch_size"></a> [sqs\_batch\_size](#input\_sqs\_batch\_size) | Max size of SQS batch to lambda | `number` | `10` | no |
| <a name="input_sqs_batching_window"></a> [sqs\_batching\_window](#input\_sqs\_batching\_window) | Maximum batching window for SQS with lambda | `number` | `0` | no |
| <a name="input_sqs_max_concurrency"></a> [sqs\_max\_concurrency](#input\_sqs\_max\_concurrency) | Maximum concurrency of batches for SQS with lambda | `number` | `100` | no |
| <a name="input_tfe_org_name"></a> [tfe\_org\_name](#input\_tfe\_org\_name) | Terraform Cloud organization name | `string` | `"ByItsCover"` | no |

## Outputs

No outputs.
<!-- END_TF_DOCS -->