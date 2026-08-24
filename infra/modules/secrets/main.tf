variable "app_name" {
  type = string
}

variable "environment" {
  type = string
}

resource "aws_secretsmanager_secret" "app" {
  name = "${var.app_name}/${var.environment}/config"
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    openai_api_key = "REPLACE_ME"
    database_url   = "REPLACE_ME"
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}

output "secret_arn" {
  value = aws_secretsmanager_secret.app.arn
}
