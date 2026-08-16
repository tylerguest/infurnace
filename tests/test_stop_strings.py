import unittest
from infurnace.tokenizer import check_stop_strings


class TestStopStrings(unittest.TestCase):
    def test_no_match(self):
        self.assertEqual(check_stop_strings("hello", 5, ["world"], True), (False, "", 0))

    def test_match_include_in_output(self):
        self.assertEqual(
            check_stop_strings("hello world", 5, ["world"], True),
            (True, "world", len("hello world")),
        )

    def test_match_exclude_from_output(self):
        self.assertEqual(
            check_stop_strings("hello world", 5, ["world"], False),
            (True, "world", 6),
        )

    def test_match_at_index(self):
        self.assertEqual(check_stop_strings("abSTOPcd", 2, ["STOP"], True), (True, "STOP", 8))

    def test_first_match_wins(self):
        matched, stop, trunc = check_stop_strings("xyZZzz", 2, ["ZZ", "zz"], False)
        self.assertEqual(matched, True)
        self.assertEqual(stop, "ZZ")
        self.assertEqual(trunc, 2)
