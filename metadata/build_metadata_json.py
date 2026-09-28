import os
import csv
import json

def parse_number(val):
    if val is None or val == "":
        return None
    try:
        val_str = str(val).replace(',', '').strip()
        if '.' in val_str:
            return float(val_str)
        return int(val_str)
    except ValueError:
        return val

def generate_metadata():
    # Base directory (repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    
    metadata_dir = os.path.join(project_root, "metadata")
    csv_path = os.path.join(metadata_dir, "jewellery_metadata.csv")
    output_json_path = os.path.join(project_root, "metadata.json")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading CSV metadata from: {csv_path}")
    
    metadata_dict = {}
    total_processed = 0

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get("Image Path", "").strip().replace('\\', '/')
            if not raw_path:
                continue

            category = row.get("Category", "").strip()
            product_type = row.get("Product Type", "").strip()
            metal_color = row.get("Metal Color", "").strip()
            gold_purity = row.get("Gold Purity", "").strip()
            shape = row.get("Shape", "").strip()
            type_of_wear = row.get("Type of Wear", "").strip()
            stone_type = row.get("Stone Type", "").strip()
            stone_cut = row.get("Stone Cut", "").strip()
            description = row.get("Description", "").strip()

            # Build feature tags list
            features = []
            for attr in [category, product_type, metal_color, gold_purity, shape, type_of_wear, stone_type, stone_cut]:
                if attr and attr.lower() not in ["not applicable", "no stone", "unknown", "none"]:
                    if attr not in features:
                        features.append(attr)

            meta_entry = {
                "category": category,
                "product_type": product_type,
                "metal_color": metal_color,
                "gold_purity": gold_purity,
                "shape": shape,
                "type_of_wear": type_of_wear,
                "stone_type": stone_type,
                "stone_cut": stone_cut,
                "size_type": row.get("Size Type", "").strip(),
                "size": parse_number(row.get("Size", "")),
                "gross_weight_g": parse_number(row.get("Gross Weight (g)", "")),
                "net_weight_g": parse_number(row.get("Net Weight (g)", "")),
                "stone_weight_g": parse_number(row.get("Stone Weight (g)", "")),
                "gold_rate_inr_g": parse_number(row.get("Gold Rate Used (INR/g)", "")),
                "estimated_cost_inr": parse_number(row.get("Estimated Cost (INR)", "")),
                "gst_rate": row.get("GST Rate", "").strip(),
                "gst_amount_inr": parse_number(row.get("GST Amount (INR)", "")),
                "total_price_incl_gst_inr": parse_number(row.get("Total Price incl. GST (INR)", "")),
                "description": description,
                "info": description,
                "confidence_score": parse_number(row.get("Confidence Score", "")),
                "split": row.get("Split", "").strip(),
                "features": features
            }

            # Strategy: Index by multiple key formats for seamless lookups
            # 1. Raw CSV relative path (e.g., train/Bangles/020_0001.jpg)
            metadata_dict[raw_path] = meta_entry
            
            # 2. Path prefixed with dataset/ (e.g., dataset/train/Bangles/020_0001.jpg)
            if not raw_path.startswith("dataset/"):
                dataset_rel_path = f"dataset/{raw_path}"
                metadata_dict[dataset_rel_path] = meta_entry
            else:
                dataset_rel_path = raw_path
                
            # 3. Absolute path on host machine
            abs_path = os.path.abspath(os.path.join(project_root, dataset_rel_path)).replace('\\', '/')
            metadata_dict[abs_path] = meta_entry
            
            total_processed += 1

    print(f"Processed {total_processed} items into {len(metadata_dict)} indexed path entries.")
    
    # Save output JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

    print(f"Successfully created metadata JSON at: {output_json_path}")

if __name__ == "__main__":
    generate_metadata()
