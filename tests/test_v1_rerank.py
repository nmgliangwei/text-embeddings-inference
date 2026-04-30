import unittest
import requests

MODEL_IDS = [
    {
        "model_id": "bge-reranker-large",
        "env": {},
    },
    {
        "model_id": "Qwen3-Reranker-0.6B",
        "env": {
            "IS_RERANK": "1",
            "DEFAULT_PROMPT": "<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: <s1>\n<Document>: <s2>",
        },
    },
]
PORT = 3078


def _make_test_class(model_id_, port_, env_=None):
    model_name = model_id_.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")
    class_name = f"TestV1Rerank_{model_name}"

    from tests.test_utils import TestBase

    class Cls(TestBase):
        model_id = model_id_
        port = port_
        env = env_ or {}
        base_url = f"http://127.0.0.1:{port_}"

        def test_rerank_basic(self):
            payload = {
                "query": "什么是机器学习？",
                "documents": [
                    "机器学习是人工智能的一个分支。",
                    "今天天气真好。",
                    "深度学习是机器学习的子领域。",
                ],
            }
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertIn("results", data)
            self.assertEqual(len(data["results"]), 3)
            for item in data["results"]:
                self.assertIn("index", item)
                self.assertIn("relevance_score", item)
                self.assertIsInstance(item["relevance_score"], float)

        def test_rerank_with_top_n(self):
            payload = {
                "query": "什么是深度学习？",
                "documents": [
                    "深度学习是机器学习的子领域。",
                    "今天天气真好。",
                    "深度学习使用神经网络。",
                    "我喜欢吃苹果。",
                ],
                "top_n": 2,
            }
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertEqual(len(data["results"]), 2)

        def test_rerank_return_documents(self):
            payload = {
                "query": "什么是机器学习？",
                "documents": [
                    "机器学习是人工智能的一个分支。",
                    "今天天气真好。",
                ],
                "return_documents": True,
            }
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            for item in data["results"]:
                self.assertIn("document", item)
                self.assertIsNotNone(item["document"])
                self.assertIn("text", item["document"])

        def test_rerank_single_document(self):
            payload = {
                "query": "什么是机器学习？",
                "documents": [
                    "机器学习是人工智能的一个分支。",
                ],
            }
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)

            data = resp.json()
            self.assertEqual(len(data["results"]), 1)
            self.assertEqual(data["results"][0]["index"], 0)

        def test_rerank_empty_documents_rejected(self):
            payload = {
                "query": "什么是机器学习？",
                "documents": [],
            }
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
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


        def test_rerank_precision(self):
            """测试同一实例连续请求的结果一致性"""
            # 测试数据
            test_query = "什么是机器学习？"
            test_documents = [
                "机器学习是人工智能的一个分支。",
                "今天天气真好。",
                "深度学习是机器学习的子领域。",
                "我喜欢吃苹果。",
                "The capital of China is Beijing.",
                "Gravity is a force that attracts two bodies towards each other."
            ]
            payload = {
                "query": test_query,
                "documents": test_documents,
            }


            # 获取当前重排序结果
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            results_prev = data["results"]

            # 获取当前重排序结果
            resp = requests.post(f"{self.base_url}/v1/rerank", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            results_current = data["results"]

            # 比较两次结果的relevance_score
            self.assertEqual(len(results_prev), len(results_current),
                             msg="重排序结果数量不一致")

            for i, (result_prev, result_curr) in enumerate(zip(results_prev, results_current)):
                # 验证index一致
                self.assertEqual(result_prev["index"], result_curr["index"],
                                 msg=f"文档{i}的索引不一致")
                # 验证relevance_score一致
                score_prev = result_prev["relevance_score"]
                score_curr = result_curr["relevance_score"]
                self.assertAlmostEqual(score_prev, score_curr, delta=0.01,
                                       msg=f"文档{i}的重排序分数不一致，期望{score_prev}，实际{score_curr}")
                print(f"文档{i} (index={result_curr['index']}) 的relevance_score: {score_curr}")

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
