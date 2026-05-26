# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    XML_DIR = "S3DBSP-main/performanceTest"
    OUTPUT_CSV = "permin_dataset_processing/milp_labels/items_to_label.csv"
    
    xml_files = sorted(Path(XML_DIR).glob("BSP_100_O6_*.xml"))
    
    rows = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        orders_node = root.find('orders')
        if orders_node is None: continue
        
        for order in orders_node.findall('order'):
            order_id = order.get('id')
            for item in order.findall('item'):
                p = float(item.find('p').text)
                q = float(item.find('q').text)
                r = float(item.find('r').text)
                rows.append({
                    'instance_name': xml_file.name,
                    'order_id': order_id,
                    'p': p,
                    'q': q,
                    'r': r
                })
                
    df = pd.DataFrame(rows)
    # Ensure directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Exported {len(df)} items to {OUTPUT_CSV}")
    logger.info("This file can now be read by the Java Gurobi wrapper to generate labels.")

if __name__ == "__main__":
    main()
