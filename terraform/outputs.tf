output "embed_server_url" {
  value = aws_apigatewayv2_stage.embed_stage.invoke_url
}
