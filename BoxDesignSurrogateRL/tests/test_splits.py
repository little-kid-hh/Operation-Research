from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from box_design_surrogate.splits import materialize_splits, split_counts, stable_hash


class SplitMaterializationTest(unittest.TestCase):
    def test_split_counts_assigns_remainder_to_test(self) -> None:
        self.assertEqual(split_counts(10, 0.6, 0.2, 0.2), (6, 2, 2))
        self.assertEqual(split_counts(11, 0.6, 0.2, 0.2), (6, 2, 3))

    def test_materialize_splits_preserves_original_order_ids(self) -> None:
        source = """<orders_root><norder>6</norder><orders>
<order id="10"><item id="0"><p>1</p><q>1</q><r>1</r></item></order>
<order id="20"><item id="0"><p>2</p><q>1</q><r>1</r></item></order>
<order id="30"><item id="0"><p>3</p><q>1</q><r>1</r></item></order>
<order id="40"><item id="0"><p>4</p><q>1</q><r>1</r></item></order>
<order id="50"><item id="0"><p>5</p><q>1</q><r>1</r></item></order>
<order id="60"><item id="0"><p>6</p><q>1</q><r>1</r></item></order>
</orders></orders_root>"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_xml = tmp_path / "toy.xml"
            source_xml.write_text(source, encoding="utf-8")

            manifest = materialize_splits(source_xml=source_xml, out_dir=tmp_path / "splits", seed=7)

            self.assertEqual(manifest["counts"], {"train": 3, "dev": 1, "test": 2})
            self.assertTrue(Path(manifest["manifest_path"]).exists())
            self.assertTrue(Path(manifest["assignments_csv"]).exists())

            all_ids = []
            for split_name, split_file in manifest["split_files"].items():
                root = ET.parse(split_file).getroot()
                ids = [order.get("id") for order in root.find("orders").findall("order")]
                self.assertEqual(int(root.findtext("norder")), len(ids))
                self.assertEqual(len(ids), manifest["counts"][split_name])
                all_ids.extend(ids)

            self.assertEqual(sorted(all_ids), ["10", "20", "30", "40", "50", "60"])

    def test_stable_hash_is_deterministic(self) -> None:
        self.assertEqual(stable_hash(1, "42"), stable_hash(1, "42"))
        self.assertNotEqual(stable_hash(1, "42"), stable_hash(2, "42"))


if __name__ == "__main__":
    unittest.main()
