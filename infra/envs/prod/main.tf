terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "support-assistant-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "support-assistant"
      Environment = "prod"
      ManagedBy   = "terraform"
    }
  }
}

variable "vpc_id" {
  type = string
}

variable "subnet_ids" {
  type = list(string)
}

variable "container_image" {
  type = string
}

variable "certificate_arn" {
  type        = string
  description = "ACM certificate ARN for HTTPS listener"
}

module "secrets" {
  source      = "../../modules/secrets"
  app_name    = "support-assistant"
  environment = "prod"
}

module "rds" {
  source         = "../../modules/rds"
  app_name       = "support-assistant"
  environment    = "prod"
  vpc_id         = var.vpc_id
  subnet_ids     = var.subnet_ids
  instance_class = "db.r6g.large"
}

module "ecs" {
  source          = "../../modules/ecs"
  app_name        = "support-assistant"
  environment     = "prod"
  container_image = var.container_image
  vpc_id          = var.vpc_id
  subnet_ids      = var.subnet_ids
  secrets_arn     = module.secrets.secret_arn
  cpu             = 1024
  memory          = 2048
  desired_count   = 2
}

module "monitoring" {
  source       = "../../modules/monitoring"
  app_name     = "support-assistant"
  environment  = "prod"
  cluster_name = "support-assistant-prod"
  service_name = "support-assistant-prod"
}

output "alb_dns" {
  value = module.ecs.alb_dns
}

output "db_endpoint" {
  value     = module.rds.endpoint
  sensitive = true
}
