#!/usr/bin/env python3
"""
Monitor HR-VITON Progress
"""

import os
import time
import psutil
import glob

def check_python_processes():
    """Check for Python processes that might be running HR-VITON"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
        try:
            if proc.info['name'] == 'python.exe':
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return python_processes

def check_output_files():
    """Check if any output files have been generated"""
    output_dirs = [
        './output/small_test/test/unpaired/generator/output/',
        './output/small_test/test/unpaired/generator/grid/'
    ]
    
    files_found = []
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = glob.glob(os.path.join(output_dir, '*.png'))
            files_found.extend(files)
    
    return files_found

def monitor_progress():
    """Monitor the progress of HR-VITON"""
    print("HR-VITON Progress Monitor")
    print("=" * 40)
    
    while True:
        # Check Python processes
        processes = check_python_processes()
        high_cpu_processes = [p for p in processes if p['cpu_percent'] > 10]
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Status Update:")
        
        if high_cpu_processes:
            print(f"✅ HR-VITON is running! Found {len(high_cpu_processes)} active Python processes")
            for proc in high_cpu_processes:
                print(f"   - PID {proc['pid']}: CPU {proc['cpu_percent']:.1f}%, Memory {proc['memory_info'].rss / 1024 / 1024:.1f}MB")
        else:
            print("❌ No active HR-VITON processes found")
        
        # Check output files
        output_files = check_output_files()
        if output_files:
            print(f"✅ Generated {len(output_files)} output files:")
            for file in output_files[-5:]:  # Show last 5 files
                print(f"   - {os.path.basename(file)}")
        else:
            print("⏳ No output files generated yet...")
        
        # Check if output directory exists
        if os.path.exists('./output'):
            print("✅ Output directory created")
        else:
            print("⏳ Output directory not yet created")
        
        print("\nPress Ctrl+C to stop monitoring")
        print("-" * 40)
        
        time.sleep(10)  # Update every 10 seconds

if __name__ == "__main__":
    try:
        monitor_progress()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. HR-VITON may still be running in the background.")
        print("Check the output directory for results when processing is complete.") 