# Operations Runbook

## Deploy

### Prerequisites

- AWS CLI configured with appropriate credentials
- Terraform >= 1.5 installed
- Docker image pushed to ECR

### Standard Deploy

```bash
# Build and push image
docker build -t support-assistant .
docker tag support-assistant:latest <ECR_URI>:latest
docker push <ECR_URI>:latest

# Deploy infrastructure (from infra/envs/<env>)
cd infra/envs/prod
terraform init
terraform plan -var="container_image=<ECR_URI>:latest" \
               -var="vpc_id=<VPC_ID>" \
               -var="subnet_ids=[\"<SUBNET_A>\",\"<SUBNET_B>\"]" \
               -var="certificate_arn=<CERT_ARN>"
terraform apply
```

### ECS Service Update (image-only)

```bash
aws ecs update-service \
  --cluster support-assistant-prod \
  --service support-assistant-prod \
  --force-new-deployment
```

## Rollback

### ECS Rollback (last known good image)

```bash
# List recent task definitions
aws ecs list-task-definitions \
  --family-prefix support-assistant-prod \
  --sort DESC --max-items 5

# Roll back to previous revision
aws ecs update-service \
  --cluster support-assistant-prod \
  --service support-assistant-prod \
  --task-definition support-assistant-prod:<PREVIOUS_REVISION>
```

### Terraform Rollback

```bash
cd infra/envs/prod
git checkout HEAD~1 -- .
terraform apply
```

## Key Rotation

### OpenAI API Key

1. Generate a new key in the OpenAI dashboard.
2. Update Secrets Manager:
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id support-assistant/prod/config \
     --secret-string '{"openai_api_key":"<NEW_KEY>","database_url":"<EXISTING_URL>"}'
   ```
3. Force ECS redeployment to pick up the new secret:
   ```bash
   aws ecs update-service \
     --cluster support-assistant-prod \
     --service support-assistant-prod \
     --force-new-deployment
   ```
4. Revoke the old key in OpenAI dashboard.

### Database Password

RDS uses managed master password (auto-rotated). If manual rotation is needed:

```bash
aws rds modify-db-instance \
  --db-instance-identifier support-assistant-prod \
  --rotate-master-user-password
```

Update the `database_url` in Secrets Manager with the new credentials, then force ECS redeployment.

## Re-index Knowledge Base

```bash
# Run ingestion against the running service
curl -X POST https://<ALB_DNS>/ingest \
  -H "X-API-Key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"source_uri": "s3://support-docs/latest/"}'
```

For a full re-index, clear the vector store first:

```bash
# Connect to RDS
psql $DATABASE_URL -c "TRUNCATE chunks;"

# Then trigger ingestion
curl -X POST https://<ALB_DNS>/ingest ...
```

## Incident Response

### High Latency (p95 > 4s)

1. Check CloudWatch dashboard: `support-assistant-prod`.
2. Inspect ECS service events:
   ```bash
   aws ecs describe-services \
     --cluster support-assistant-prod \
     --services support-assistant-prod \
     --query 'services[0].events[:5]'
   ```
3. Check application logs:
   ```bash
   aws logs tail /ecs/support-assistant-prod --since 30m
   ```
4. If CPU is high, scale out:
   ```bash
   aws ecs update-service \
     --cluster support-assistant-prod \
     --service support-assistant-prod \
     --desired-count 4
   ```
5. If the issue is upstream (OpenAI), check their status page.

### High Error Rate (5xx > 10/5min)

1. Check application logs for stack traces.
2. Verify database connectivity:
   ```bash
   aws rds describe-db-instances \
     --db-instance-identifier support-assistant-prod \
     --query 'DBInstances[0].DBInstanceStatus'
   ```
3. Check Secrets Manager access (IAM role permissions).
4. If a bad deploy, rollback (see Rollback section).

### Database Issues

1. Check RDS metrics in CloudWatch (connections, CPU, storage).
2. If connection exhaustion, restart tasks:
   ```bash
   aws ecs update-service \
     --cluster support-assistant-prod \
     --service support-assistant-prod \
     --force-new-deployment
   ```

## Kill Switch

To immediately stop serving traffic without destroying infrastructure:

```bash
# Scale to zero
aws ecs update-service \
  --cluster support-assistant-prod \
  --service support-assistant-prod \
  --desired-count 0
```

To restore:

```bash
aws ecs update-service \
  --cluster support-assistant-prod \
  --service support-assistant-prod \
  --desired-count 2
```

## Health Check

```bash
curl https://<ALB_DNS>/healthz
```

Expected response: `{"status": "ok"}`

## Monitoring

- **Dashboard**: CloudWatch → Dashboards → `support-assistant-prod`
- **Alarms**: CloudWatch → Alarms → filter `support-assistant-prod`
- **Logs**: CloudWatch → Log groups → `/ecs/support-assistant-prod`
