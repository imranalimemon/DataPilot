import unittest
from streamlit.testing.v1 import AppTest

class TestAppFilters(unittest.TestCase):
    def setUp(self):
        import os
        app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
        self.at = AppTest.from_file(app_path)
        self.at.run(timeout=15)
        # Find the button that contains 'Load demo data'
        demo_btn = [b for b in self.at.button if 'Load demo data' in b.label]
        if demo_btn:
            demo_btn[0].click().run(timeout=15)

    def test_default_load(self):
        self.assertTrue(any('Analysis filters' in m.value for m in self.at.subheader))
        for m in self.at.metric:
            if m.label == 'Total revenue':
                self.assertEqual(m.value, '6,242.50')

    def test_empty_matches_state(self):
        prod_chk = [c for c in self.at.checkbox if c.label == 'Filter by Product'][0]
        prod_chk.set_value(True).run(timeout=15)
        
        ms = [m for m in self.at.multiselect if m.label == 'Select products'][0]
        ms.set_value([]).run(timeout=15)
        
        errs = [e.value for e in self.at.error]
        self.assertTrue(any('No rows match the active filters.' in e for e in errs))
        
        for m in self.at.metric:
            if m.label == 'Total revenue':
                self.assertEqual(m.value, '0.00')

    def test_no_auto_trigger_examples(self):
        ex_btn = [b for b in self.at.button if 'What is my total revenue?' in b.label]
        self.assertTrue(len(ex_btn) > 0)
        ex_btn[0].click().run(timeout=15)
        # Should populate text input but NOT run the qa since 'Ask' was not clicked
        ti = [t for t in self.at.text_input if 'Ask a question' in t.label][0]
        self.assertEqual(ti.value, 'What is my total revenue?')
        # Ensure qa result is not present
        mds = [m.value for m in self.at.markdown]
        self.assertFalse(any('Supporting data — Sales Summary' in m for m in mds))

if __name__ == '__main__':
    unittest.main()
