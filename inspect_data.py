"""Comprehensive data inspection script for Amazon ML Challenge 2026."""

import os
import sys
import json
from collections import Counter, defaultdict

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

FILES = [
    ("data/train/train_source1.tsv", "S1-", ["entity_id", "business_name", "business_address", "country"]),
    ("data/train/train_source2.tsv", "S2-", ["entity_id", "business_name", "business_address", "country"]),
    ("data/train/train_source3.tsv", "S3-", ["entity_id", "business_name", "business_address", "country"]),
    ("data/test/test_source1.tsv", "S1-", ["entity_id", "business_name", "business_address", "country"]),
    ("data/test/test_source2.tsv", "S2-", ["entity_id", "business_name", "business_address", "country"]),
    ("data/test/test_source3.tsv", "S3-", ["entity_id", "business_name", "business_address", "country"]),
]

GT_FILE = "data/train/train_ground_truth.tsv"

def inspect_source_file(path, expected_prefix, expected_cols):
    print(f"\n=======================================================")
    print(f"Inspecting: {path}")
    print(f"=======================================================")
    
    file_size = os.path.getsize(path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            print("ERROR: File is empty!")
            return {}
        
        # Check delimiter
        has_tab = "\t" in header
        has_comma = "," in header
        print(f"Delimiter check: Tab present={has_tab}, Comma present={has_comma}")
        
        cols = [c.strip() for c in header.rstrip("\r\n").split("\t")]
        print(f"Actual columns: {cols}")
        print(f"Expected columns: {expected_cols}")
        cols_match = cols == expected_cols
        print(f"Schema match: {cols_match}")
        
        total_rows = 0
        missing_counts = defaultdict(int)
        prefix_violations = 0
        sample_violations = []
        country_counts = Counter()
        
        # Duplicate detection: track seen entity IDs
        # To avoid high memory on 5M IDs, we can store 64-bit hashes
        seen_hashes = set()
        duplicate_count = 0
        sample_duplicates = []
        
        sample_names = []
        sample_addrs = []
        
        for idx, line in enumerate(f):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            
            if len(parts) < len(cols):
                # pad with empty strings if trailing tab was omitted
                parts.extend([""] * (len(cols) - len(parts)))
            elif len(parts) > len(cols):
                # unexpected extra column
                if total_rows <= 5:
                    print(f"Warning: row {total_rows} has {len(parts)} columns instead of {len(cols)}")
            
            eid = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else ""
            addr = parts[2].strip() if len(parts) > 2 else ""
            country = parts[3].strip() if len(parts) > 3 else ""
            
            # Check prefix
            if not eid.startswith(expected_prefix):
                prefix_violations += 1
                if len(sample_violations) < 5:
                    sample_violations.append(eid)
            
            # Check duplicates
            h = hash(eid)
            if h in seen_hashes:
                duplicate_count += 1
                if len(sample_duplicates) < 5:
                    sample_duplicates.append(eid)
            else:
                seen_hashes.add(h)
            
            # Check missing values
            if not eid:
                missing_counts["entity_id"] += 1
            if not name:
                missing_counts["business_name"] += 1
            if not addr:
                missing_counts["business_address"] += 1
            if not country:
                missing_counts["country"] += 1
                
            country_counts[country if country else "<EMPTY>"] += 1
            
            # Collect sample records
            if len(sample_names) < 10 and name:
                sample_names.append((eid, name))
            if len(sample_addrs) < 10 and addr:
                sample_addrs.append((eid, addr))
                
            if total_rows % 1000000 == 0:
                print(f"  Processed {total_rows:,} rows...")
                
    print(f"Total rows (excluding header): {total_rows:,}")
    print(f"Duplicate entity IDs: {duplicate_count:,} (Sample: {sample_duplicates})")
    print(f"Prefix violations (not starting with {expected_prefix}): {prefix_violations:,} (Sample: {sample_violations})")
    print("Missing values:")
    for c in cols:
        cnt = missing_counts[c]
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        print(f"  {c:20s}: {cnt:,} missing ({pct:.2f}%)")
    print("Country distribution:")
    for c, cnt in country_counts.most_common():
        pct = (cnt / total_rows * 100) if total_rows > 0 else 0
        print(f"  {c:20s}: {cnt:,} ({pct:.2f}%)")
        
    return {
        "path": path,
        "size_bytes": file_size,
        "total_rows": total_rows,
        "cols": cols,
        "duplicate_count": duplicate_count,
        "prefix_violations": prefix_violations,
        "missing_counts": dict(missing_counts),
        "country_counts": dict(country_counts),
        "sample_names": sample_names,
        "sample_addrs": sample_addrs,
    }


def inspect_ground_truth(path):
    print(f"\n=======================================================")
    print(f"Inspecting Ground Truth: {path}")
    print(f"=======================================================")
    
    file_size = os.path.getsize(path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        cols = [c.strip() for c in header.rstrip("\r\n").split("\t")]
        print(f"Actual columns: {cols}")
        expected_cols = ["source1_entity_id", "matched_entity_ids"]
        print(f"Expected columns: {expected_cols}")
        print(f"Schema match: {cols == expected_cols}")
        
        total_rows = 0
        empty_matches = 0
        non_empty_matches = 0
        match_counts = []
        
        s2_count = 0
        s3_count = 0
        s1_self_matches = 0
        other_prefix_matches = 0
        
        seen_s1 = set()
        dup_s1 = 0
        
        sample_pairs = []
        
        for idx, line in enumerate(f):
            total_rows += 1
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0].strip()
            rest = parts[1].strip() if len(parts) > 1 else ""
            
            h = hash(s1_id)
            if h in seen_s1:
                dup_s1 += 1
            else:
                seen_s1.add(h)
                
            if not rest:
                empty_matches += 1
            else:
                non_empty_matches += 1
                matched = [m.strip() for m in rest.split(",") if m.strip()]
                match_counts.append(len(matched))
                
                for m in matched:
                    if m.startswith("S2-"):
                        s2_count += 1
                    elif m.startswith("S3-"):
                        s3_count += 1
                    elif m.startswith("S1-"):
                        s1_self_matches += 1
                    else:
                        other_prefix_matches += 1
                        
                if len(sample_pairs) < 10:
                    sample_pairs.append((s1_id, matched[:4]))
                    
            if total_rows % 500000 == 0:
                print(f"  Processed {total_rows:,} ground truth rows...")
                
    import numpy as np
    match_arr = np.array(match_counts) if match_counts else np.array([0])
    
    print(f"Total S1 entities in GT: {total_rows:,}")
    print(f"Duplicate S1 rows: {dup_s1:,}")
    print(f"Singletons (0 matches): {empty_matches:,} ({empty_matches / total_rows * 100:.2f}%)")
    print(f"Entities with >= 1 match: {non_empty_matches:,} ({non_empty_matches / total_rows * 100:.2f}%)")
    print(f"Total matched target links: {sum(match_counts):,}")
    print(f"  S2 links: {s2_count:,} ({s2_count / sum(match_counts) * 100:.2f}%)")
    print(f"  S3 links: {s3_count:,} ({s3_count / sum(match_counts) * 100:.2f}%)")
    print(f"  S1 self-matches: {s1_self_matches:,}")
    print(f"  Invalid prefix matches: {other_prefix_matches:,}")
    print(f"Match count distribution (for non-singletons):")
    print(f"  Min: {match_arr.min()}, Max: {match_arr.max()}")
    print(f"  Mean: {match_arr.mean():.2f}, Median: {np.median(match_arr):.1f}, Std: {match_arr.std():.2f}")
    print(f"  25th percentile: {np.percentile(match_arr, 25):.1f}, 75th percentile: {np.percentile(match_arr, 75):.1f}")
    
    print("\nSample ground truth entries:")
    for s1, mlist in sample_pairs[:5]:
        print(f"  {s1} -> {mlist}")
        
    return {
        "path": path,
        "total_rows": total_rows,
        "cols": cols,
        "empty_matches": empty_matches,
        "non_empty_matches": non_empty_matches,
        "match_stats": {
            "total_links": int(sum(match_counts)),
            "s2_links": s2_count,
            "s3_links": s3_count,
            "s1_self_matches": s1_self_matches,
            "min": int(match_arr.min()),
            "max": int(match_arr.max()),
            "mean": float(match_arr.mean()),
            "median": float(np.median(match_arr)),
            "p25": float(np.percentile(match_arr, 25)),
            "p75": float(np.percentile(match_arr, 75)),
        },
        "sample_pairs": sample_pairs,
    }

if __name__ == "__main__":
    results = {}
    for path, pfx, cols in FILES:
        res = inspect_source_file(path, pfx, cols)
        results[path] = res
        
    gt_res = inspect_ground_truth(GT_FILE)
    results[GT_FILE] = gt_res
    
    with open("docs/inspection_results.json", "w", encoding="utf-8") as f:
        # serialize simple json
        json.dump(results, f, default=str, indent=2)
    print("\nData inspection complete! Results saved to docs/inspection_results.json")
