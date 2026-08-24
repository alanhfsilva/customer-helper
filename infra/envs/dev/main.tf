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
    key    = "dev/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = "support-assistant"
      Environment = "dev"
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

module "secrets" {
  source      = "../../modules/secrets"
  app_name    = "support-assistant"
  environment = "dev"
}

module "rds" {
  source         = "../../modules/rds"
  app_name       = "support-assistant"
  environment    = "dev"
  vpc_id         = var.vpc_id
  subnet_ids     = var.subnet_ids
  instance_class = "db.t3.micro"
}

module "ecs" {
  source          = "../../modules/ecs"
  app_name        = "support-assistant"
  environment     = "dev"
  container_image = var.container_image
  vpc_id          = var.vpc_id
  subnet_ids      = var.subnet_ids
  secrets_arn     = module.secrets.secret_arn
  cpu             = 256
  memory          = 512
  desired_count   = 1
}

module "monitoring" {
  source       = "../../modules/monitoring"
  app_name     = "support-assistant"
  environment  = "dev"
  cluster_name = "support-assistant-dev"
  service_name = "support-assistant-dev"
}

output "alb_dns" {
  value = module.ecs.alb_dns
}

output "db_endpoint" {
  value = module.rds.endpoint
}
