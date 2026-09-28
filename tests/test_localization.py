import json
import unittest
from pathlib import Path
from src.api.localization import translate, class_label, HINDI

ROOT = Path(__file__).resolve().parents[1]


class LocalizationTests(unittest.TestCase):
    def test_frontend_and_report_share_translations(self):
        frontend = json.loads((ROOT / 'frontend/src/locales/hi.json').read_text(encoding='utf-8'))
        self.assertEqual(frontend, HINDI)
        self.assertFalse(any('???' in value for value in HINDI.values()))

    def test_hindi_labels_and_care_are_translated(self):
        self.assertEqual(class_label('Tomato___healthy', 'hi'), 'टमाटर - स्वस्थ')
        self.assertEqual(translate('LEAF ANALYSIS REPORT', 'hi'), 'पत्ती विश्लेषण रिपोर्ट')
        self.assertEqual(translate('Healthy green leaves.', 'en'), 'Healthy green leaves.')
        info = json.loads((ROOT / 'data/disease_info.json').read_text())
        for record in info.values():
            for key in ['symptoms', 'treatment', 'prevention']:
                value = record.get(key)
                if value:
                    self.assertNotEqual(translate(value, 'hi'), value, (key, value))


if __name__ == '__main__':
    unittest.main()
