#!/usr/bin/env python3
"""
Script to parse training log files (*.log) and extract validation accuracy metrics.

Usage:
  parse_val_accuracy_logs.py [--log_dir LOG_DIR] [--threshold THRESHOLD] [--output OUTPUT_CSV]

Options:
  --log_dir    Directory where .log files are located (default: current directory)
  --threshold  Only include runs whose best validation accuracy >= THRESHOLD (default: no filter)
  --output     Path to output CSV file (default: val_accuracy_summary.csv)
"""
import os
import re
import glob
import sys
import csv
import argparse


def parse_log_file(path):
    """
    Parse a single log file to extract (epoch, val_accuracy) pairs.
    """
    epoch_re = re.compile(r'^Epoch\s+(\d+)/')
    val_re = re.compile(r'val_accuracy[:=]\s*([0-9\.]+)')
    metrics = []
    current_epoch = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            m = epoch_re.match(line)
            if m:
                current_epoch = int(m.group(1))
                continue
            if current_epoch is not None:
                m2 = val_re.search(line)
                if m2:
                    val_acc = float(m2.group(1))
                    metrics.append((current_epoch, val_acc))
    return metrics


def main():
    p = argparse.ArgumentParser(description="Extract best and last validation accuracy from training logs")
    p.add_argument('--log_dir', default='.', help='Directory containing .log files')
    p.add_argument('--threshold', type=float, default=None,
                   help='Filter: only include runs with best val_accuracy >= threshold')
    p.add_argument('--output', default='val_accuracy_summary.csv',
                   help='Output CSV file path')
    args = p.parse_args()

    log_dir = args.log_dir
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    if not log_files:
        print(f"No .log files found in {log_dir}", file=sys.stderr)
        sys.exit(1)

    results = []
    for log_path in log_files:
        metrics = parse_log_file(log_path)
        if not metrics:
            continue
        # best and last
        best_epoch, best_acc = max(metrics, key=lambda x: x[1])
        last_epoch, last_acc = metrics[-1]
        results.append((os.path.basename(log_path), best_epoch, best_acc, last_epoch, last_acc))

    # apply threshold filter
    if args.threshold is not None:
        results = [r for r in results if r[2] >= args.threshold]

    # sort by best accuracy descending
    results.sort(key=lambda x: x[2], reverse=True)

    # write CSV
    with open(args.output, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['log_file', 'best_epoch', 'best_val_accuracy', 'last_epoch', 'last_val_accuracy'])
        for row in results:
            writer.writerow(row)
    print(f"Summary written to {args.output}")

    # print summary table
    print(f"{'Log File':<40} {'Epoch':>5} {'BestAcc':>8}")
    for log_file, be, ba, le, la in results:
        print(f"{log_file:<40} {be:5d} {ba:8.4f}")


if __name__ == '__main__':
    main()