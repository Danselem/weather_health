# Legacy MLflow Infrastructure (Archived)

This stack is retained for reference only. It was used to run MLflow on a
single EC2 instance with a publicly reachable RDS instance.

**Do not use this for new deployments.** Use [`infra-aws/`](../infra-aws/),
which has private subnets, Secrets Manager, remote state, and an ECS service.

The legacy stack has been limited where possible:

- RDS is no longer publicly accessible.
- SSH is restricted to `allowed_ssh_cidr` (default: the VPC CIDR).

It still uses local state and is not part of the supported CI/CD path.
