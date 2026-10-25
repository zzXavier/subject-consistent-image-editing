import unittest
from src.data import items,prepare


class ProductInputs(unittest.TestCase):
    def test_usable_subjects_have_distinct_views_and_disjoint_splits(self):
        chosen = [r for r in items() if r["id"]!="B083V9C8D2"]
        development={r["id"] for r in chosen if r["split"]=="dev"}
        held_out={r["id"] for r in chosen if r["split"]=="test"}
        self.assertEqual((len(development),len(held_out)),(2,2))
        self.assertFalse(development & held_out)
        for row in chosen:
            original,_,image_hash=prepare(row)
            ref,_,ref_hash=prepare(row,ref=True)
            self.assertNotEqual(image_hash,ref_hash)
            self.assertEqual(original.size,ref.size)
