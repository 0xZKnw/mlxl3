import unittest

from mlxl3_exl3.download import DownloadError, validate


class ValidateTests(unittest.TestCase):
    def test_accepts(self):
        self.assertEqual(validate("0xzknw/LFM3-1.7B-EXL3", "/4bpw/", None), "4bpw")
        self.assertEqual(validate("a/b", "", "main"), "")
        self.assertEqual(validate("a/b", "x/y", "0123abc"), "x/y")

    def test_rejects(self):
        for repo, folder, revision in [("ab", "", None), ("a/b/c", "", None), ("../b", "", None),
                                       ("a/b", "../x", None), ("a/b", "x/../y", None),
                                       ("a/b", "C:\\x", None), ("a/b", "x\\y", None),
                                       ("a/b", "", "../main"), ("a/b", "", "")]:
            with self.subTest(repo=repo, folder=folder, revision=revision):
                with self.assertRaises(DownloadError):
                    validate(repo, folder, revision)


if __name__ == "__main__":
    unittest.main()
