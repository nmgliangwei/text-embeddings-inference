import json
import os
import unittest
import requests

from tests.test_utils import _compute_cosine_distance

MODEL_IDS = [
    {
        "model_id": "bge-large-zh-v1.5",
        "env": {},
    },
    {
        "model_id": "bge-m3",
        "env": {},
    },
    {
        "model_id": "Qwen3-Embedding-0.6B",
        "env": {},
    },
]
PORT = 3068


def _make_test_class(model_id_, port_, env_=None):
    model_name = model_id_.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")
    class_name = f"TestV1Embeddings_{model_name}"

    from tests.test_utils import TestBase
    
    class Cls(TestBase):
        model_id = model_id_
        port = port_
        env = env_ or {}
        base_url = f"http://127.0.0.1:{port_}"

        def test_single_input(self):
            payload = {
                "input": "你好，世界",
                "model": self.model_id,
            }
            resp = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertEqual(data["object"], "list")
            self.assertEqual(len(data["data"]), 1)
            self.assertEqual(data["data"][0]["object"], "embedding")
            self.assertIsInstance(data["data"][0]["embedding"], list)
            self.assertGreater(len(data["data"][0]["embedding"]), 0)
            self.assertEqual(data["data"][0]["index"], 0)
            self.assertIn("model", data)
            self.assertIn("usage", data)
            self.assertIn("prompt_tokens", data["usage"])
            self.assertGreater(data["usage"]["prompt_tokens"], 0)

        def test_batch_input(self):
            payload = {
                "input": ["你好", "世界", "测试文本"],
            }
            resp = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertEqual(data["object"], "list")
            self.assertEqual(len(data["data"]), 3)
            for i, item in enumerate(data["data"]):
                self.assertEqual(item["object"], "embedding")
                self.assertIsInstance(item["embedding"], list)
                self.assertGreater(len(item["embedding"]), 0)
                self.assertEqual(item["index"], i)

        def test_embedding_dimension_consistency(self):
            payload_single = {
                "input": "维度一致性测试",
            }
            resp_single = requests.post(
                f"{self.base_url}/v1/embeddings", json=payload_single
            )
            self.assertEqual(resp_single.status_code, 200)
            dim_single = len(resp_single.json()["data"][0]["embedding"])

            payload_batch = {
                "input": ["维度一致性测试", "第二条文本"],
            }
            resp_batch = requests.post(
                f"{self.base_url}/v1/embeddings", json=payload_batch
            )
            self.assertEqual(resp_batch.status_code, 200)
            for item in resp_batch.json()["data"]:
                self.assertEqual(len(item["embedding"]), dim_single)

        def test_with_dimensions(self):
            payload = {
                "input": "维度截断测试",
                "dimensions": 256,
            }
            resp = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                self.assertEqual(len(data["data"][0]["embedding"]), 256)

        def test_empty_input_rejected(self):
            payload = {
                "input": "",
            }
            resp = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            self.assertIn(resp.status_code, [400, 422])

        def test_health_endpoint(self):
            resp = requests.get(f"{self.base_url}/health")
            self.assertEqual(resp.status_code, 200)

        def test_info_endpoint(self):
            resp = requests.get(f"{self.base_url}/info")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("model_id", data)
            self.assertIn("model_type", data)

        def test_embedding_precision(self):
            """测试同一实例连续请求的结果一致性"""
            # 测试文本
            test_texts = ["测试文本1", "测试文本2", "这是一个更长的测试文本，用于验证模型的精度稳定性",
                          "The capital of China is Beijing.",
                          "Gravity is a force that attracts two bodies towards each other."]
            payload = {"input": test_texts}

            # 获取当前嵌入结果
            resp1 = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            self.assertEqual(resp1.status_code, 200)

            pre_embeddings = [item["embedding"] for item in resp1.json()["data"]]

            # 获取当前嵌入结果
            resp1 = requests.post(f"{self.base_url}/v1/embeddings", json=payload)
            self.assertEqual(resp1.status_code, 200)

            current_embeddings = [item["embedding"] for item in resp1.json()["data"]]

            # 计算余弦距离
            for i, (emb_prev, emb_curr) in enumerate(zip(pre_embeddings, current_embeddings)):
                cosine_dist = _compute_cosine_distance(emb_prev, emb_curr)
                self.assertAlmostEqual(cosine_dist, 1.0, delta=0.01,
                                       msg=f"文本{i}的两次嵌入结果余弦距离不足，期望接近1.0，实际为{cosine_dist}")
                print(f"文本{i}的余弦距离: {cosine_dist}")


    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    return Cls


for _i, _model_cfg in enumerate(MODEL_IDS):
    _port = PORT + _i
    _cls = _make_test_class(_model_cfg["model_id"], _port, env_=_model_cfg.get("env"))
    globals()[_cls.__name__] = _cls

del _i, _model_cfg, _port, _cls

if __name__ == "__main__":
    unittest.main()
