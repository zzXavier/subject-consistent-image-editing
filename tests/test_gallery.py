import unittest
import numpy as np
from PIL import Image
from src.experiment import subject_signature


class GalleryTests(unittest.TestCase):
    def test_background_cannot_change_foreground_identity_score(self):
        red = Image.new("RGB",(4,4),"red")
        mixed = red.copy()
        mixed.paste("blue",(2,0,4,4))
        mask = Image.new("L",(4,4),0)
        mask.paste(255,(0,0,2,4))
        d1 = subject_signature(red,mask)
        d2 = subject_signature(mixed,mask)
        self.assertTrue(np.allclose(d1,d2))
        self.assertAlmostEqual(float(d1 @ d2),1.)
