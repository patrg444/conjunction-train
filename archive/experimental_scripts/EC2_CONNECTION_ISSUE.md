# EC2 Connection Issue Documentation

## Current State

We're encountering persistent SSH authentication issues with the AWS g5.2xlarge instance (`i-03d733fa99127153a` at IP `52.90.38.179`):

- The instance is running (confirmed via `aws ec2 describe-instances`)
- The instance was launched with a key pair named "new-key" (confirmed in instance metadata)
- The PEM key file exists locally at `~/Downloads/new-key.pem` with correct permissions (600)
- All deployment scripts have been updated to use this key path
- SSH connection attempts are consistently failing with "Permission denied (publickey)"

## Attempted Solutions

1. ✅ Updated script paths to use correct key file location (`~/Downloads/new-key.pem`)
2. ✅ Set proper restrictive permissions on key file (`chmod 600 ~/Downloads/new-key.pem`)
3. ✅ Tried direct SSH with correct key and StrictHostKeyChecking disabled

## Recommended Next Steps

To resolve this issue, we need to investigate the following possibilities:

1. **Key Mismatch**: The local PEM file might not correspond to the AWS key pair.
   - Solution: Generate a new key pair in AWS, download the new PEM file, and restart instance with new key

2. **Instance Configuration**: The EC2 instance might have SSH configuration issues.
   - Solution: Review AWS console logs for EC2 instance, check if SSH service is running properly

3. **Security Group Settings**: The EC2 security group might be blocking connections.
   - Solution: Verify security group allows SSH (port 22) from your IP address

4. **AWS EC2 Instance User Data**: Check if custom user data is modifying SSH configuration.
   - Solution: Verify instance user data in EC2 console

## Immediate Action Needed

Before continuing with the deployment and training scripts, we must first establish successful SSH connectivity to the instance. 

```bash
# For testing direct connection only
ssh -v -i ~/Downloads/new-key.pem ubuntu@52.90.38.179

# If you need to rebuild the instance with a different key:
aws ec2 stop-instances --instance-ids i-03d733fa99127153a
aws ec2 create-key-pair --key-name emotion-key-new --query 'KeyMaterial' --output text > ~/Downloads/emotion-key-new.pem
chmod 600 ~/Downloads/emotion-key-new.pem
# Terminate and relaunch instance with new key
```

Once SSH connection is established, you can continue with the workflow:

```bash
./deploy_to_aws_g5.sh  # For deployment
./agentic_train_laughter_g5.sh  # For full automated workflow
