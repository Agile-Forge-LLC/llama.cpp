import unittest

import gguf


class TestQwen4ExpMtpMapping(unittest.TestCase):
    def test_mtp_block_and_trunk_tensor_names(self):
        tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, 49)
        names = {
            "model.layers.48.eh_proj.weight": "blk.48.nextn.eh_proj.weight",
            "model.layers.48.hyper_connection_mixer.hc_norm.weight": "blk.48.nextn.hc_head_norm.weight",
            "model.layers.48.hyper_connection_mixer.input_mix_weight_down.weight": "blk.48.nextn.hc_head_down.weight",
            "model.layers.48.hyper_connection_mixer.input_mix_weight_up.weight": "blk.48.nextn.hc_head_up.weight",
            "model.layers.48.self_attn.indexer.q_layernorm.weight": "blk.48.indexer.q_norm.weight",
            "model.layers.0.self_attn.indexer.q_layernorm.weight": "blk.0.indexer.q_norm.weight",
        }
        for source, expected in names.items():
            with self.subTest(source=source):
                self.assertEqual(tensor_map.get_name(source, (".weight",)), expected)


if __name__ == "__main__":
    unittest.main()
