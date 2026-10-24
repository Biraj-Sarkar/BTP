import unittest
from extractor.training.annotation_tool import display_to_mask, fit_geometry

class AnnotationMappingTests(unittest.TestCase):
    def test_landscape_corners_and_center(self):
        g=fit_geometry((600,900),(900,600))
        self.assertEqual(display_to_mask(0,0,(600,900),g),(0,0))
        self.assertEqual(display_to_mask(450,300,(600,900),g),(450,300))
        self.assertEqual(display_to_mask(899,599,(600,900),g),(899,599))

    def test_portrait_letterbox_rejected_and_points_mapped(self):
        g=fit_geometry((1200,900),(900,600))
        self.assertIsNone(display_to_mask(100,300,(1200,900),g))
        self.assertEqual(display_to_mask(450,300,(1200,900),g),(450,600))
        self.assertEqual(display_to_mask(674,599,(1200,900),g),(898,1198))

    def test_current_image_geometry(self):
        g=fit_geometry((4096,3072),(900,600))
        self.assertIsNone(display_to_mask(224,300,(4096,3072),g))
        self.assertEqual(display_to_mask(450,300,(4096,3072),g),(1536,2048))
        self.assertEqual(display_to_mask(674,599,(4096,3072),g),(3065,4089))

if __name__=='__main__': unittest.main()
