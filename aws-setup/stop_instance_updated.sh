#!/bin/bash
# Script to stop or terminate the EC2 instance

INSTANCE_ID="i-0dd2f787db00b205f"

echo "Do you want to stop or terminate the instance?"
echo "1) Stop instance (can be restarted later, storage charges apply)"
echo "2) Terminate instance (permanent deletion, no further charges)"
read -p "Enter choice [1-2]: " choice

case $choice in
    1)
        echo "Stopping instance $INSTANCE_ID..."
        aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
        echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids $INSTANCE_ID"
        ;;
    2)
        echo "Terminating instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
        echo "Instance terminated."
        ;;
    *)
        echo "Invalid choice, no action taken."
        ;;
esac
