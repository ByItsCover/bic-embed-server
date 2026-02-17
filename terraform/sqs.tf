locals {
  sqs_arn = data.terraform_remote_state.bic_infra.outputs.sqs_arn
}

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = sqs_arn
  function_name    = aws_lambda_function.embed_function.arn
  batch_size       = 10

  scaling_config {
    maximum_concurrency = 100
  }
}
