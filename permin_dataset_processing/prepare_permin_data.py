# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Determine container dimensions from filename or default
        # In S3DBSP, container is often ID 89 (120x120x120) or derived from problem context
        # For evaluation, we'll assume the standard vehicle/bin used in our MILP: 60x25x30
        # unless the paper specified otherwise.
        
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
    PACKAGES_PATH = "S3DBSP-main/performanceTest/packages.txt"
    XML_DIR = "S3DBSP-main/performanceTest"
    OUTPUT_CSV = "permin_dataset_processing/processed_features/permin_base40_features.csv"
    
    parser = PerminParser(PACKAGES_PATH)
    all_features = []
    
    # Iterate through XML instances (let's start with O6 baseline)
    xml_files = sorted(Path(XML_DIR).glob("BSP_100_O6_*.xml"))
    
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
    df_final.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Saved {len(df_final)} instances to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
