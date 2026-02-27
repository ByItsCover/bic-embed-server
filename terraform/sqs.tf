locals {
  sqs_arn = data.terraform_remote_state.bic_infra.outputs.sqs_arn
}

resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn                   = local.sqs_arn
  function_name                      = aws_lambda_function.embed_function.arn
  batch_size                         = var.sqs_batch_size
  maximum_batching_window_in_seconds = var.sqs_batching_window

  scaling_config {
    maximum_concurrency = var.sqs_max_concurrency
  }
}
