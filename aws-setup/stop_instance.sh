#!/bin/bash
# This script stops or terminates the EC2 instance

echo "Do you want to stop or terminate the instance?"
echo "1) Stop instance (can be restarted later, storage charges apply)"
echo "2) Terminate instance (permanent deletion, no further charges)"
read -p "Enter choice [1-2]: " choice

case  in
    1)
        echo "Stopping instance i-0dd2f787db00b205f..."
        aws ec2 stop-instances --instance-ids "i-0dd2f787db00b205f"
        echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids i-0dd2f787db00b205f"
        ;;
    2)
        echo "Terminating instance i-0dd2f787db00b205f..."
        aws ec2 terminate-instances --instance-ids "i-0dd2f787db00b205f"
        echo "Instance terminated."
        ;;
    *)
        echo "Invalid choice, no action taken."
        ;;
esac
