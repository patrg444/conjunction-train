#!/bin/bash
# Directly stop or terminate instance without menu

INSTANCE_ID="i-0dd2f787db00b205f"

if [[ $1 == "stop" ]]; then
    echo "Stopping instance $INSTANCE_ID..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
    echo "Instance stopped. To restart it later use: aws ec2 start-instances --instance-ids $INSTANCE_ID"
    exit 0
elif [[ $1 == "terminate" ]]; then
    echo "Terminating instance $INSTANCE_ID..."
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
    echo "Instance terminated."
    exit 0
else
    echo "Usage: $0 [stop|terminate]"
    echo "  stop - Stop the instance (can be restarted later, storage charges apply)"
    echo "  terminate - Terminate the instance (permanent deletion, no further charges)"
    exit 1
fi
