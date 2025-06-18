#!/bin/bash
# Package the project for Elastic Beanstalk deployment
zip -r launch-darkly-demo.zip . -x "*.git*" "*__pycache__*" "venv/*" ".env" ".DS_Store"
