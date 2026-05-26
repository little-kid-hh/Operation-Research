# Permin Dataset Labeling Pipeline

This directory contains the scripts needed to generate Ground Truth labels for the Permin (2022) 3D Bin Packing dataset using Gurobi.

## Workflow

1.  **Run `permin_dataset_processing/export_items_for_milp.py` (Already Done)**
    This script extracts the dimensions of items from the XML files into `permin_dataset_processing/milp_labels/items_to_label.csv`. This provides a flat structure that is easy to load in Java.

2.  **Move to a machine with Gurobi Installed**
    If the current machine does not have a Gurobi Academic License installed and activated, copy the entire project (including `MILP_3DBPP` and `permin_dataset_processing`) to the machine with Gurobi.

3.  **Run the Java Label Generator**
    Navigate to the `MILP_3DBPP` directory and use Maven to compile and run the wrapper class.

    ```bash
    cd MILP_3DBPP
    mvn clean compile
    mvn exec:java -Dexec.mainClass="org.example.GeneratePerminLabels"
    ```

    *Note: Ensure your `pom.xml` properly points to your local Gurobi `.jar` file.*

4.  **Merge Results**
    The Java script will output the feasibility results to `permin_dataset_processing/milp_labels/ground_truth_labels.csv`. It contains labels for both `2-orientations` and `6-orientations`.
    You can then merge these labels back with the `permin_base40_features.csv` to evaluate the ML models.
