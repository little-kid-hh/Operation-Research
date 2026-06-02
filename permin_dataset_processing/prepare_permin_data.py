# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OR2023_BPP_XML_DIR = Path("or2023_bpp_data/xml")
OR2023_BPP_PACKAGES = Path("or2023_bpp_data/packages.txt")
S3DBSP_PATH_MARKER = "S3DBSP-main"


def reject_bsp_derived_path(path: Path, allow_bsp_derived_data: bool) -> None:
    if allow_bsp_derived_data:
        return
    if S3DBSP_PATH_MARKER.lower() in str(path).replace("\\", "/").lower():
        raise ValueError(
            f"{path} is a S3DBSP/stochastic-BSP data path. "
            "Use the Fontaine & Minner OR 2023 3D-BPP data path instead."
        )


class PerminParser:
    def __init__(self, packages_file):
        self.bins = self._load_bins(packages_file)

    def _load_bins(self, file_path):
        """Load bin types from packages.txt: ID, L, W, H"""
        bins = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    # ID: [L, W, H]
                    bins[int(parts[0])] = [float(parts[1]), float(parts[2]), float(parts[3])]
        return bins

    def parse_xml(self, xml_file):
        """Parse order XML and return a list of items for each order."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # The container/package dimensions are attached later when the candidate
        # package is chosen for the BPP feasibility task.
        
        all_orders = []
        orders_node = root.find('orders')
        if orders_node is None:
            return []

        for order in orders_node.findall('order'):
            order_id = order.get('id')
            items = []
            for item in order.findall('item'):
                items.append({
                    'dim_l': float(item.find('p').text),
                    'dim_m': float(item.find('q').text),
                    'dim_s': float(item.find('r').text),
                })
            
            # Sort dims to ensure s <= m <= l consistency
            for itm in items:
                dims = sorted([itm['dim_l'], itm['dim_m'], itm['dim_s']])
                itm['dim_s'], itm['dim_m'], itm['dim_l'] = dims
            
            all_orders.append({
                'order_id': order_id,
                'items': items
            })
        return all_orders

def calculate_base40_features(items, v_l=60.0, v_w=25.0, v_h=30.0):
    """
    Calculate the 40 base features expected by the ML models.
    Ref: research/loadability_model_pack_20260526_v2/models/svm_base40/linear_svm_20260427_192130.json
    """
    v_cap = v_l * v_w * v_h
    n_items = len(items)
    if n_items == 0: return None
    
    df_items = pd.DataFrame(items)
    df_items['vol'] = df_items['dim_l'] * df_items['dim_m'] * df_items['dim_s']
    
    total_vol = df_items['vol'].sum()
    
    res = {}
    res['sku_counts'] = n_items
    res['sku_average_volume'] = df_items['vol'].mean()
    res['sku_length_var'] = df_items['dim_l'].var() if n_items > 1 else 0
    res['sku_width_var'] = df_items['dim_m'].var() if n_items > 1 else 0
    res['sku_height_var'] = df_items['dim_s'].var() if n_items > 1 else 0
    res['sku_length_avg'] = df_items['dim_l'].mean()
    res['sku_width_avg'] = df_items['dim_m'].mean()
    res['sku_height_avg'] = df_items['dim_s'].mean()
    
    # Aspect ratios and other derived features
    df_items['asr'] = df_items['dim_l'] / df_items['dim_s'].replace(0, 1e-9)
    res['max_asr'] = df_items['asr'].max()
    
    res['vehicle_length'] = v_l
    res['vehicle_width'] = v_w
    res['vehicle_height'] = v_h
    res['spare_capacity'] = v_cap - total_vol
    
    # Concentration (vol dominance)
    res['sku_concentration'] = df_items['vol'].max() / total_vol if total_vol > 0 else 0
    
    res['sku_min_length'] = df_items['dim_l'].min()
    res['sku_max_length'] = df_items['dim_l'].max()
    res['sku_std_length'] = df_items['dim_l'].std() if n_items > 1 else 0
    res['sku_min_width'] = df_items['dim_m'].min()
    res['sku_max_width'] = df_items['dim_m'].max()
    res['sku_std_width'] = df_items['dim_m'].std() if n_items > 1 else 0
    res['sku_min_height'] = df_items['dim_s'].min()
    res['sku_max_height'] = df_items['dim_s'].max()
    res['sku_std_height'] = df_items['dim_s'].std() if n_items > 1 else 0
    
    # L, H, W Ratios
    res['l_to_L_ratio_avg'] = (df_items['dim_l'] / v_l).mean()
    res['l_to_L_ratio_min'] = (df_items['dim_l'] / v_l).min()
    res['l_to_L_ratio_max'] = (df_items['dim_l'] / v_l).max()
    res['l_to_L_ratio_std'] = (df_items['dim_l'] / v_l).std() if n_items > 1 else 0
    
    res['h_to_H_ratio_avg'] = (df_items['dim_s'] / v_h).mean()
    res['h_to_H_ratio_min'] = (df_items['dim_s'] / v_h).min()
    res['h_to_H_ratio_max'] = (df_items['dim_s'] / v_h).max()
    res['h_to_H_ratio_std'] = (df_items['dim_s'] / v_h).std() if n_items > 1 else 0
    
    res['w_to_W_ratio_avg'] = (df_items['dim_m'] / v_w).mean()
    res['w_to_W_ratio_min'] = (df_items['dim_m'] / v_w).min()
    res['w_to_W_ratio_max'] = (df_items['dim_m'] / v_w).max()
    res['w_to_W_ratio_std'] = (df_items['dim_m'] / v_w).std() if n_items > 1 else 0
    
    # Footprint ratios
    res['wl_to_vehicle_wl_avg'] = (df_items['dim_l'] * df_items['dim_m'] / (v_l * v_w)).mean()
    res['wl_to_vehicle_wl_min'] = (df_items['dim_l'] * df_items['dim_m'] / (v_l * v_w)).min()
    res['wl_to_vehicle_wl_max'] = (df_items['dim_l'] * df_items['dim_m'] / (v_l * v_w)).max()
    res['wl_to_vehicle_wl_std'] = (df_items['dim_l'] * df_items['dim_m'] / (v_l * v_w)).std() if n_items > 1 else 0
    res['wl_to_vehicle_wl_total'] = (df_items['dim_l'] * df_items['dim_m']).sum() / (v_l * v_w)

    return res

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--packages-path", type=Path, default=OR2023_BPP_PACKAGES)
    cli.add_argument("--xml-dir", type=Path, default=OR2023_BPP_XML_DIR)
    cli.add_argument("--xml-glob", default="*.xml")
    cli.add_argument(
        "--output-csv",
        type=Path,
        default=Path("permin_dataset_processing/processed_features/or2023_bpp_base40_features.csv"),
    )
    cli.add_argument(
        "--allow-bsp-derived-data",
        action="store_true",
        help="Legacy escape hatch: allow S3DBSP/stochastic-BSP paths for audits only.",
    )
    args = cli.parse_args()
    reject_bsp_derived_path(args.packages_path, args.allow_bsp_derived_data)
    reject_bsp_derived_path(args.xml_dir, args.allow_bsp_derived_data)
    
    parser = PerminParser(args.packages_path)
    all_features = []
    
    xml_files = sorted(args.xml_dir.glob(args.xml_glob))
    if not xml_files:
        raise ValueError(f"No XML files matched {args.xml_glob!r} in {args.xml_dir}")
    
    for xml_file in xml_files:
        logger.info(f"Processing {xml_file.name}...")
        orders = parser.parse_xml(xml_file)
        
        for order in orders:
            feats = calculate_base40_features(order['items'])
            if feats:
                feats['instance_source'] = xml_file.name
                feats['order_id'] = order['order_id']
                all_features.append(feats)
                
    df_final = pd.DataFrame(all_features)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(args.output_csv, index=False)
    logger.info(f"Saved {len(df_final)} instances to {args.output_csv}")

if __name__ == "__main__":
    main()
