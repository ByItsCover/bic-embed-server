data "aws_iam_policy_document" "policy-document" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}


resource "aws_iam_policy" "function_logging_policy" {
  name   = "function-logging-policy"
  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [
      {
        Action : [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ],
        Effect : "Allow",
        Resource : "arn:aws:logs:*:*:*"
      }
    ]
  })
}


resource "aws_iam_role" "api_function_role" {
  name = "lambda_iam_role"

  assume_role_policy = data.aws_iam_policy_document.policy-document.json
}

resource "aws_iam_role_policy_attachment" "basic" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.api_function_role.name
}

resource "aws_iam_role_policy_attachment" "function_logging_policy_attachment" {
  policy_arn = aws_iam_policy.function_logging_policy.arn
  role       = aws_iam_role.api_function_role.id
}
