#!/bin/bash
# EC2 bootstrap: install and start Docker on Amazon Linux 2023.
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
