# Quick Start Guide: AWS GPU Training

This guide provides the fastest way to get your emotion recognition model training on AWS GPU instances.

## Step 1: Package Your Data (Run Locally)

```bash
cd /Users/patrickgloria/conjunction-train
cd aws-setup
./prepare-data.sh
```

This will create an `aws-temp` directory with the necessary files.

## Step 2: Launch GPU Training (Run Locally)

```bash
cd /Users/patrickgloria/conjunction-train
cd aws-setup
./aws_gpu_fallback.sh
```

This script will:
1. Try multiple GPU instance types, starting with high-performance options (g4dn.24xlarge, g5.24xlarge, p4d.24xlarge)
2. Fall back to smaller instances if the high-performance ones aren't available
3. Set up the environment with GPU-specific optimizations
4. Start training automatically
5. Create utility scripts for monitoring and management

## Step 3: Monitor Training

Once training starts, you can monitor progress:

```bash
# Check training logs
./check_progress.sh

# Monitor GPU usage
./monitor_gpu.sh
```

## Step 4: Download Results

After training completes, download your models:

```bash
./download_results.sh
```

## Step 5: Stop the Instance

When finished, stop or terminate your instance to avoid costs:

```bash
./stop_instance.sh
```

## Performance Expectations

- **g4dn.24xlarge**: ~15-25x speedup over local training (~$7.67/hour)
- **g5.24xlarge**: ~20-30x speedup over local training (~$9.36/hour)
- **p4d.24xlarge**: ~40-60x speedup over local training (~$32.77/hour)

Your 50-epoch training could complete in just minutes instead of hours!

## Troubleshooting

If you encounter any errors:

1. **No instances available**: Your account may have service limits. Try requesting limit increases or using smaller instances.
2. **Authentication errors**: Double-check your AWS credentials.
3. **Connection issues**: Ensure your security group allows SSH access.
