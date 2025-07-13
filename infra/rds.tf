resource "aws_security_group" "rds_sg" {
  name   = "rds_sg"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "Postgres access from VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "default" {
  name       = "mlflow-db-subnet-group"
  subnet_ids = [
    aws_subnet.public_1.id,
    aws_subnet.public_2.id
  ]
}

resource "aws_db_instance" "mlflow_pg" {
  identifier             = "mlflow-db"
  allocated_storage       = 20
  storage_type            = "gp2"
  engine                  = "postgres"
  engine_version          = "17.5"
  instance_class          = "db.t3.micro"
  username                = "mlflow"
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.default.name
  vpc_security_group_ids  = [aws_security_group.rds_sg.id]
  publicly_accessible     = true
  skip_final_snapshot     = true

  tags = {
    Name = "mlflow-postgres"
  }
}

