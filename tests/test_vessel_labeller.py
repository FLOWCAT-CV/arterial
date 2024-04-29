import unittest
import os, shutil
from arterial.vessel_labelling.vessel_labeller import VesselLabeller

class TestCenterlineExtractor(unittest.TestCase):
    def setUp(self):
        self.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        self.mode = "extracranial_vessels"
        self.centerline_segments_array_path = os.path.join(self.case_dir, "input_test_data", "centerline_segments_array.npy")
        self.vesssel_labeller = VesselLabeller(self.case_dir, self.mode, self.centerline_segments_array_path)

    def test_init(self):
        self.assertEqual(self.vesssel_labeller.case_dir, self.case_dir)
        self.assertEqual(self.vesssel_labeller.mode, self.mode)
        self.assertEqual(self.vesssel_labeller.centerline_segments_array_path, self.centerline_segments_array_path)
        self.assertIsNone(self.vesssel_labeller.centerline_segments_array)
        self.assertIsNone(self.vesssel_labeller.segments_graph)
        self.assertIsNone(self.vesssel_labeller.segments_graph_pred)
        self.assertEqual(self.vesssel_labeller.segments_graph_path, os.path.join(self.case_dir, self.mode, "segments_graph.pickle"))
        self.assertEqual(self.vesssel_labeller.segments_graph_pred_path, os.path.join(self.case_dir, self.mode, "segments_graph_pred.pickle"))
        self.assertEqual(self.vesssel_labeller.segments_graph_plot_path, os.path.join(self.case_dir, self.mode, "segments_graph.png"))
        self.assertEqual(self.vesssel_labeller.segments_graph_pred_plot_path, os.path.join(self.case_dir, self.mode, "segments_graph_pred.png"))

    def test_full_pipeline(self):
        self.vesssel_labeller.build_segments_graph(save=False)
        self.assertIsNotNone(self.vesssel_labeller.centerline_segments_array)
        self.assertIsNotNone(self.vesssel_labeller.segments_graph)
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_path))
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_plot_path))
        self.vesssel_labeller.centerline_segments_array = None
        self.vesssel_labeller.segments_graph = None
        self.assertIsNone(self.vesssel_labeller.centerline_segments_array)
        self.assertIsNone(self.vesssel_labeller.segments_graph)
        
        self.vesssel_labeller.build_segments_graph(save=True)
        self.assertIsNotNone(self.vesssel_labeller.centerline_segments_array)
        self.assertIsNotNone(self.vesssel_labeller.segments_graph)
        self.assertTrue(os.path.exists(self.vesssel_labeller.segments_graph_path))
        self.assertTrue(os.path.exists(self.vesssel_labeller.segments_graph_plot_path))

        self.vesssel_labeller.predict_vessel_types(ensemble=False, save=False)
        self.assertIsNotNone(self.vesssel_labeller.segments_graph_pred)
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_pred_path))
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_pred_plot_path))
        self.vesssel_labeller.segments_graph_pred = None
        self.assertIsNone(self.vesssel_labeller.segments_graph_pred)
        
        self.vesssel_labeller.predict_vessel_types(ensemble=True, save=False)
        self.assertIsNotNone(self.vesssel_labeller.segments_graph_pred)
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_pred_path))
        self.assertFalse(os.path.exists(self.vesssel_labeller.segments_graph_pred_plot_path))
        self.vesssel_labeller.segments_graph_pred = None
        self.assertIsNone(self.vesssel_labeller.segments_graph_pred)

        self.vesssel_labeller.predict_vessel_types(ensemble=True, save=True)
        self.assertIsNotNone(self.vesssel_labeller.segments_graph_pred)
        self.assertTrue(os.path.exists(self.vesssel_labeller.segments_graph_path))
        self.assertTrue(os.path.exists(self.vesssel_labeller.segments_graph_plot_path))

        self.vesssel_labeller.centerline_segments_array = None
        self.vesssel_labeller.segments_graph = None

        self.vesssel_labeller.load_centerline_segments_array()
        self.assertIsNotNone(self.vesssel_labeller.centerline_segments_array)
        
        self.vesssel_labeller.load_segments_graph()
        self.assertIsNotNone(self.vesssel_labeller.segments_graph)

    @classmethod
    def tearDownClass(cls):
        # Remove all the files generated during the tests
        cls.case_dir = os.path.join(os.path.dirname(__file__), "test_data")
        for filename in os.listdir(cls.case_dir):
            if filename not in ["input_test_data", "output"]:
                if os.path.isfile(os.path.join(cls.case_dir, filename)):
                    os.remove(os.path.join(cls.case_dir, filename))
                elif os.path.isdir(os.path.join(cls.case_dir, filename)):
                    shutil.rmtree(os.path.join(cls.case_dir, filename))

if __name__ == '__main__':
    unittest.main()