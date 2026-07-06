# text-embeddings-inference(TEI)

> English | [中文](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/OVERVIEW.zh.md)

## Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/text-embeddings-inference/issues)
  - [text-embeddings-inference Code](https://gitcode.com/Ascend/text-embeddings-inference)
  - [text-embeddings-inference API Reference](https://huggingface.github.io/text-embeddings-inference/)
  - [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)
  - [Community](https://www.hiascend.com/)

## Overview
This image is based on text embedding inference and is adapted for Huawei NPU cards to provide service interfaces such as embedding and reranker. The interfaces are compatible with the open-source text embedding inference and support models such as embedding, reranker, and Sequence Classification. By default, this image runs as the regular user `HwHiAiUser`.

## Dockerfile link corresponding to mirror tags

### Tag Naming Convention

Tags follow this pattern:

```bash
<TEI_version>-<chip_series>-<os>-<python_version>
```

| Field            | Example Values                  | Description                       |
|------------------|---------------------------------|-----------------------------------|
| `TEI_version`    | `26.0.0`                        | text-embeddings-inference version |
| `chip_series`    | `910`, `a3`, `Atlas 300I Pro`   | Target chip family                |
| `os`             | `ubuntu22.04`, `openeuler24.03` | Base operating system             |
| `python_version` | `py3.11`                        | Python version                    |

### Docker Image Matching Table
| TEI    | CANN  | torch-npu |
|--------|-------|-----------|
| 26.0.0 | 9.0.0 | 2.9.0     |


---

## Quick Start

### Prerequisites (optional)

#### Install Driver

An Atlas NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### Running a TEI Container

Startup command for version 26.0.0 and later:

```bash
export MODEL_DIR=/home/HwHiAiUser/model
docker run -itd \
    -u <user> \
    --net host \
    --name tei_container \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -e MODEL_DIR=$MODEL_DIR \
    -v $MODEL_DIR:$MODEL_DIR \
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag> --model-id $MODEL_DIR/<model_name> --hostname <listen_ip> --port <listen_port>
```

<details> <summary>Startup command for versions before 26.0.0:</summary>

```bash
export MODEL_DIR=/home/HwHiAiUser/model
docker run -itd \
    -u <user> \
    --net host \
    --name tei_container \
    -e ENABLE_BOOST=True \
    --device /dev/davinci0 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v $MODEL_DIR:/home/HwHiAiUser/model \
    -e PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256 \
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag>  /home/HwHiAiUser/model/<model_name> <listen_ip> <listen_port>
```

</details>

#### **Parameter Description**
- The `MODEL_DIR` environment variable indicates the directory on the host machine where the model is stored. Before running the container, the model weight files need to be downloaded and stored in this directory.
- `PYTORCH_NPU_ALLOC_CONF` NPU memory block allows splitting up to a limit, for more information please check the [link](https://www.hiascend.com/document/detail/zh/Pytorch/2600/comref/Envvariables/docs/zh/environment_variable_reference/PYTORCH_NPU_ALLOC_CONF.md)
- Please replace `<user>`、`<tag>`、`<model_name>`、`<listen_ip>`、`<listen_port>` with actual values according to your situation.
- Please ensure that the model files on the host machine match `<model name>`.
- The container runs by default as the regular user `HwHiAiUser`. Please make sure the HwHiAiUser has read and write permissions for the models on the host. If you need to specify a different user, add the `-u <user>` parameter when running the container.
- Please refer to the [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md) to learn other configuration parameters.

## List of supported models and explanations of specific parameters for each model

### Text Embeddings
| Model Name                                                                                              | Model Type  | Other required parameters                      |
|---------------------------------------------------------------------------------------------------------|-------------|------------------------------------------------|
| [BAAI/bge-large-zh-v1.5](https://www.modelscope.cn/models/BAAI/bge-large-zh-v1.5)                       | BERT        | -                                              |
| [BAAI/bge-m3](https://www.modelscope.cn/models/BAAI/bge-m3)                                             | XLM-RoBERTa | -                                              |
| [aspire/acge_text_embedding](https://www.modelscope.cn/models/yangjhchs/acge_text_embedding)            | BERT        | -                                              |
| [jinaai/jina-embeddings-v2-base-zh](https://www.modelscope.cn/models/jinaai/jina-embeddings-v2-base-zh) | BERT        | -e TRUST_REMOTE_CODE=1                         |
| [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://www.modelscope.cn/models/iic/gte-Qwen2-1.5B-instruct)     | Qwen2       | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false |
| [Alibaba-NLP/gte-Qwen2-7B-instruct](https://www.modelscope.cn/models/iic/gte-Qwen2-7B-instruct)         | Qwen2       | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false |
| [Qwen/Qwen3-Embedding-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)                 | Qwen3       | -                                              |
| [Qwen/Qwen3-Embedding-4B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-4B)                     | Qwen3       | -                                              |
| [Qwen/Qwen3-Embedding-8B](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-8B)                     | Qwen3       | -                                              |
| [microsoft/unixcoder-base](https://www.modelscope.cn/models/microsoft/unixcoder-base)                   | RoBERTa     | -                                              |


### Sequence Classification and Re-Ranking
| Model Name                                                                                  | Model Type  | Other required parameters                                                                                                                                                    |
|---------------------------------------------------------------------------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [BAAI/bge-reranker-large](https://www.modelscope.cn/models/BAAI/bge-reranker-large)         | XLM-RoBERTa | -                                                                                                                                                                            |
| [BAAI/bge-reranker-v2-m3](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-m3)         | XLM-RoBERTa | -                                                                                                                                                                            |
| [Qwen/Qwen3-Reranker-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-0.6B)       | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [Qwen/Qwen3-Reranker-4B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-4B)           | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [Qwen/Qwen3-Reranker-8B](https://www.modelscope.cn/models/Qwen/Qwen3-Reranker-8B)           | Qwen3       | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| [jason9693/Qwen2.5-1.5B-apeach](https://www.modelscope.cn/models/Howeee/Qwen2.5-1.5B-apeac) | Qwen2       | -                                                                                                                                                                            |


## Interface Test Example

You can test whether the functions of each model service are running properly by executing the following commands

- **Rerank model testing**:

```bash
curl 127.0.0.1:8080/v1/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "documents": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

- **Dense vector model testing**:

```bash
curl 127.0.0.1:8080/v1/embeddings \
    -X POST \
    -d '{"input":["The capital of China is Beijing."]}' \
    -H 'Content-Type: application/json'
```

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["The capital of China is Beijing."]}' \
    -H 'Content-Type: application/json'
```

- **Sequence classification model testing**:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

- **Sparse vector model testing**:

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### How to build docker image

```bash
# Clone the code in any directory on the host machine, then go into the docker directory, and get the Dockerfile. Replace {cann version} with the actual version, and replace {your_repo} with the actual image repository.
git clone https://gitcode.com/Ascend/text-embeddings-inference.git && cd text-embeddings-inference/docker

docker build --network host --build-arg CANN_VERSION={cann version} -t {your_repo}/mis-tei:latest -f Dockerfile.<chip_series>.<os> .
```

---

## Supported Hardware

| Chip Series    | Product Examples               | Architecture   |
|----------------|--------------------------------|----------------|
| Atlas 910      | Atlas 800I A2                  | ARM64 / x86_64 |
| Atlas A3       | Atlas 800I A3                  | ARM64 / x86_64 |
| Atlas 300I Pro | Atlas 300I Pro, Atlas 300V Pro | ARM64 / x86_64 |

---

## Troubleshooting

Please read the [FAQ](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/FAQ.md) section of the docs first, if you still have questions, please [submit an issue](https://gitcode.com/Ascend/text-embeddings-inference/issues).

## License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
