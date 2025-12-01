#!/bin/bash

# Exit if any command fails
set -e

# Ensure non-interactive mode for apt
export DEBIAN_FRONTEND=noninteractive

echo "🔄 Updating system packages..."
sudo apt-get update -y

echo "🐳 Installing Docker..."
sudo apt-get install -y docker.io

echo "▶ Starting & enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

echo "📦 Installing unzip & curl..."
sudo apt-get install -y unzip curl

echo "⬇ Downloading AWS CLI v2..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"

echo "📂 Unzipping AWS CLI installer..."
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/

echo "⚙ Installing AWS CLI..."
sudo /home/ubuntu/aws/install

echo "👤 Adding 'ubuntu' to docker group..."
sudo usermod -aG docker ubuntu

echo "🧹 Cleaning installation files..."
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws

echo "✅ install_dependencies.sh completed successfully."
