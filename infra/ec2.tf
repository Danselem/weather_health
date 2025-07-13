resource "aws_security_group" "ec2_sg" {
  name   = "ec2_sg"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "ubuntu" {
  ami                    = "ami-0c7217cdde317cfec"  # Ubuntu 22.04 LTS for us-east-1
  instance_type          = "t2.micro"
  subnet_id              = aws_subnet.public_1.id
  key_name               = var.ec2_key_name
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]

  tags = {
    Name = "mlflow-ubuntu-server"
  }
}
