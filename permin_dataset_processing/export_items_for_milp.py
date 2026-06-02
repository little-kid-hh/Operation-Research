# -*- coding: utf-8 -*-
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OR2023_BPP_XML_DIR = Path("or2023_bpp_data/xml")
S3DBSP_PATH_MARKER = "S3DBSP-main"


def reject_bsp_derived_path(path: Path, allow_bsp_derived_data: bool) -> None:
    if allow_bsp_derived_data:
        return
    if S3DBSP_PATH_MARKER.lower() in str(path).replace("\\", "/").lower():
        raise ValueError(
            f"{path} is a S3DBSP/stochastic-BSP data path. "
            "Use the Fontaine & Minner OR 2023 3D-BPP data path instead."
        )


def main():
    import argparse

    cli = argparse.ArgumentParser()
    cli.add_argument("--xml-dir", type=Path, default=OR2023_BPP_XML_DIR)
    cli.add_argument("--xml-glob", default="*.xml")
    cli.add_argument(
        "--output-csv",
        type=Path,
        default=Path("permin_dataset_processing/milp_labels/items_to_label.csv"),
    )
    cli.add_argument(
        "--allow-bsp-derived-data",
        action="store_true",
        help="Legacy escape hatch: allow S3DBSP/stochastic-BSP paths for audits only.",
    )
    args = cli.parse_args()
    reject_bsp_derived_path(args.xml_dir, args.allow_bsp_derived_data)
    
    xml_files = sorted(args.xml_dir.glob(args.xml_glob))
    if not xml_files:
        raise ValueError(f"No XML files matched {args.xml_glob!r} in {args.xml_dir}")
    
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
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    logger.info(f"Exported {len(df)} items to {args.output_csv}")
    logger.info("This file can now be read by the Java Gurobi wrapper to generate labels.")

if __name__ == "__main__":
    main()
