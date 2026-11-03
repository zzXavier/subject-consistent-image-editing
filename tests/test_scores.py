import unittest
from PIL import Image,ImageDraw
from src.experiment import pixel_scores
class TestScores(unittest.TestCase):
    def test_edit_and_keep_are_disjoint(self):
        src=Image.new('RGB',(16,16),'white');mask=Image.new('L',src.size);ImageDraw.Draw(mask).rectangle((4,4,9,9),fill=255)
        y=Image.composite(src,Image.new('RGB',src.size,'blue'),mask)
        self.assertEqual(pixel_scores(src,y,mask,'blue'),{'outside_edit_mae':0.0,'requested_hue_fraction':1.0})

