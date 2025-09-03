#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np


def field_stats(data):
    """Compute field statistics: Mean, Min/Max, L1/L2 norms"""
    stats = {}
    stats["mean"] = np.nanmean(data)
    stats["min"] = np.nanmin(data)
    stats["max"] = np.nanmax(data)
    stats["L1"] = np.sum(np.abs(data))
    stats["L2"] = np.sqrt(np.sum(data**2))
    stats["size"] = np.prod(data.shape)
    return stats

def load_output(in_dir: Path, field: str, file_name=None):
    """Load TRITON data from asc format"""
    if file_name is None:
        in_files = sorted(Path(in_dir).glob(f"{field}_*.out"))[-1:]
    else:
        in_files = [Path(in_dir, file_name)]
    
    if not in_files:
        raise FileNotFoundError(f"No files found for field {field}")
    
    data = np.genfromtxt(in_files[0])
    return data, in_files[0]

def main():
    parser = argparse.ArgumentParser(description="Compare TRITON runs - simplified version")
    parser.add_argument("--ref", help="Path to reference TRITON run", type=Path, required=True)
    parser.add_argument("--test", help="Path to test TRITON run", type=Path, required=True)
    parser.add_argument("--file", help="Filename suffix to load (optional)", type=str, default=None)
    parser.add_argument("--verbose", help="Print detailed statistics", action="store_true")
    
    args = parser.parse_args()
    
    ref_dir = Path(args.ref)
    test_dir = Path(args.test)
    
    if not ref_dir.exists():
        print(f"{ref_dir}: does not exist")  
        sys.exit(1)

    if not test_dir.exists():
        print(f"{test_dir}: does not exist")  
        sys.exit(1)

    test_fields = ["H", "QX", "QY"]
    stats_test = {}
    stats_ref = {}
    stats_diff = {}
    all_bfb = True
     
    for field in test_fields:
        file_name = f"{field}_{args.file}" if args.file else None
        
        try:
            test_data, test_file = load_output(test_dir, field, file_name)
            ref_data, ref_file = load_output(ref_dir, field, file_name)
        except (IndexError, FileNotFoundError):
            print(f" NO FILES FOUND FOR VARIABLE: {field}")
            continue
        
        # Calculate statistics
        stats_test[field] = field_stats(test_data)
        stats_ref[field] = field_stats(ref_data)
        stats_diff[field] = field_stats(ref_data - test_data)
        
        # Check if bit-for-bit identical  
        if np.all((ref_data - test_data) == 0):  
            print(f"✅ {field}: BIT-FOR-BIT")  
        else:  
            print(f"❌ {field}: DIFFERENCES FOUND")  
            all_bfb = False  

            # Always show difference stats in non-bit-for-bit case  
            diff_stats = stats_diff[field]  
            print(f"   ➡ Mean diff: {diff_stats['mean']:.4e}, "  
                  f"Min: {diff_stats['min']:.4e}, Max: {diff_stats['max']:.4e}, "  
                  f"L2 norm: {diff_stats['L2']:.4e}")
            
    # Print detailed statistics if requested
    if args.verbose:
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)
        
        print(f"\n{'FIELD':<8} {'STAT':<8}: {'TEST':<12} {'REF':<12}")
        print("-" * 50)
        for field in stats_test:
            for stat in stats_test[field]:
                if stat == "size":
                    print(f"{field:<8} {stat:<8}: {stats_test[field][stat]:<12} {stats_ref[field][stat]:<12}")
                else:
                    print(f"{field:<8} {stat:<8}: {stats_test[field][stat]:<12.4e} {stats_ref[field][stat]:<12.4e}")
        
        print(f"\n{'FIELD':<8} {'STAT':<8}: {'DIFFERENCE':<12}")
        print("-" * 30)
        for field in stats_diff:
            for stat in stats_diff[field]:
                if stat == "size":
                    print(f"{field:<8} {stat:<8}: {stats_diff[field][stat]:<12}")
                else:
                    print(f"{field:<8} {stat:<8}: {stats_diff[field][stat]:<12.4e}")
    
    # Final summary
    print("="*50)
    if all_bfb:
        print("ALL FIELDS ARE BIT-FOR-BIT IDENTICAL")
        sys.exit(0)
    else:
        print("NOT ALL FIELDS ARE BIT-FOR-BIT IDENTICAL")
        sys.exit(1)


if __name__ == "__main__":
    main()
