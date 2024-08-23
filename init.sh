#!/bin/bash

# Exit on any error
set -e

# Print commands as they are executed
set -x

# Create directories if they don't exist
mkdir -p /workspaces/ML4T/zips

# Unzip the files
echo "Unzipping ML4T_2024Fall.zip..."
unzip -o /workspaces/ML4T/zips/ML4T_2024Fall.zip -d /workspaces/ML4T/

echo "Unzipping assess_portfolio_2024Fall.zip..."
unzip -o /workspaces/ML4T/zips/assess_portfolio_2024Fall.zip -d /workspaces/ML4T/

# Clean up the zip files
# echo "Removing zip files..."
# rm /workspaces/ML4T/zips/ML4T_2024Fall.zip
# rm /workspaces/ML4T/zips/assess_portfolio_2024Fall.zip

# List the contents after unzipping
echo "Contents of /workspaces/ML4T/zips/ after unzipping:"
ls -la /workspaces/ML4T/zips/

conda init

bash --login

