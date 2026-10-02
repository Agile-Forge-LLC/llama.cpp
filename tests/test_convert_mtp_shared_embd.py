import unittest

from mtp_shared_embd import validate_mtp_shared_embd


class TestMtpSharedEmbd(unittest.TestCase):
    def test_qwen4exp_accepts_shared_embeddings(self):
        self.assertIsNone(validate_mtp_shared_embd("qwen4exp"))

    def test_other_mtp_architectures_reject_shared_embeddings(self):
        for arch in ("qwen3next", "qwen35", "qwen35moe"):
            with self.subTest(arch=arch):
                with self.assertRaisesRegex(ValueError, rf"--mtp-shared-embd.*{arch}.*qwen4exp"):
                    validate_mtp_shared_embd(arch)


if __name__ == "__main__":
    unittest.main()
