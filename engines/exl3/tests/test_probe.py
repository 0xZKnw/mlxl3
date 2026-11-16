import unittest

from mlxl3_exl3.probe import in_wsl, no_gpu_hints, wsl_report


class ProbeTests(unittest.TestCase):
    def test_wsl_detection(self):
        self.assertTrue(in_wsl("Linux version 6.6.87.2-microsoft-standard-WSL2 (root@host)"))
        self.assertTrue(in_wsl("Linux version 4.4.0-19041-Microsoft"))
        self.assertFalse(in_wsl("Linux version 6.8.0-45-generic (buildd@lcy02)"))

    def test_wsl_report_reads_paths_and_env(self):
        present = {"/dev/dxg", "/usr/lib/wsl/lib/libdxcore.so", "/opt/rocm/lib/librocdxg.so"}
        report = wsl_report(exists=present.__contains__, environ={"HSA_ENABLE_DXG_DETECTION": "1"})
        self.assertEqual(report, {"dxg_device": True, "dxcore": True,
                                  "rocdxg": "/opt/rocm/lib/librocdxg.so", "dxg_detection": True})
        empty = wsl_report(exists=lambda _: False, environ={})
        self.assertEqual(empty, {"dxg_device": False, "dxcore": False, "rocdxg": None, "dxg_detection": False})

    def test_hints_name_each_missing_piece(self):
        missing = no_gpu_hints("rocm", {"dxg_device": False, "dxcore": False, "rocdxg": None, "dxg_detection": False})
        text = "\n".join(missing)
        for needle in ("/dev/dxg", "libdxcore", "librocdxg", "HSA_ENABLE_DXG_DETECTION", "Adrenalin", "RX 6000"):
            self.assertIn(needle, text)
        complete = no_gpu_hints("rocm", {"dxg_device": True, "dxcore": True, "rocdxg": "/x", "dxg_detection": True})
        self.assertEqual(len(complete), 2, "only driver and GPU support remain")
        self.assertIn("nvidia-smi", "\n".join(no_gpu_hints("cuda", None)))
        self.assertIn("reinstall", no_gpu_hints("cpu", None)[0])
