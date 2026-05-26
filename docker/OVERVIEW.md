# text-embeddings-inference(TEI)

> English | [中文](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/OVERVIEW.zh.md)

## Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/text-embeddings-inference/issues)
  - [text-embeddings-inference Code](https://gitcode.com/Ascend/text-embeddings-inference)
  - [text-embeddings-inference API Reference](https://huggingface.github.io/text-embeddings-inference/)
  - [Image Repository](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)
  - [Community](https://www.hiascend.com/)

## Supported Tags and Dockerfile Links

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

### CANN 9.0.0 + 26.0.0 TEI Image

| Tag                                 | Dockerfile                                                                                                    | Image Content |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------|---------------|
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.910b.ubuntu)    | cann + TEI    |
| `26.0.0-910b-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.910b.openeuler) | cann + TEI    |
| `26.0.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.a3.ubuntu)      | cann + TEI    |
| `26.0.0-a3-openeuler24.03-py3.11`   | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.a3.openeuler)   | cann + TEI    |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.310p.ubuntu)    | cann + TEI    |
| `26.0.0-310p-openeuler24.03-py3.11` | [Dockerfile](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docker/Dockerfile.310p.openeuler) | cann + TEI    |

---

## Quick Start

### Prerequisites (optional)

#### Install Driver

An Atlas NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### Running a TEI Container

```bash
export MODEL_DIR="/home/HwHiAiUser/model"
docker run \
    -u <user> \
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
    -p <host_port>:8000 \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag> --model-id $MODEL_DIR/<model name> --port 8000
```

#### **Parameter Description**

- The `MODEL_DIR` environment variable specifies the directory on the host machine where the model is stored. TEI inside the container will load the model from this directory.
- Please replace `<user>`, `<host_port>`, `<tag>`, `<model name>` with actual values according to your situation.
- Please ensure that the model files on the host machine match `<model name>`.
- The container runs by default as the regular user `HwHiAiUser`. Please make sure the HwHiAiUser has read and write permissions for the models on the host. If you need to specify a different user, add the `-u <user>` parameter when running the container.
- Please refer to the [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md) to learn other configuration parameters.

## List of supported models and explanations of specific parameters for each model

| Model Name                          | Description                    | Other required parameters                                                                                                                                                    |
|-------------------------------------|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BAAI/bge-large-zh-v1.5              | Dense Vector Model             | -                                                                                                                                                                            |
| BAAI/bge-m3                         | Dense and sparse vector models | -                                                                                                                                                                            |
| aspire/acge_text_embedding          | Dense Vector Model             | -                                                                                                                                                                            |
| BAAI/bge-reranker-large             | Ranking Model                  | -                                                                                                                                                                            |
| BAAI/bge-reranker-v2-m3             | Ranking Model                  | -                                                                                                                                                                            |
| jinaai/jina-embeddings-v2-base-zh   | Dense Vector Model             | -e TRUST_REMOTE_CODE=1                                                                                                                                                       |
| Alibaba-NLP/gte-Qwen2-1.5B-instruct | Dense Vector Model             | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false                                                                                                                               |
| Alibaba-NLP/gte-Qwen2-7B-instruct   | Dense Vector Model             | -e TRUST_REMOTE_CODE=1 <br/>-e IS_CAUSAL=false                                                                                                                               |
| Qwen/Qwen3-Embedding-0.6B(4B、8B)    | Dense Vector Model             | -                                                                                                                                                                            |
| Qwen/Qwen3-Reranker-0.6B(4B、8B)     | Ranking Model                  | -e IS_RERANK=1 <br/> -e DEFAULT_PROMPT="\<Instruct\>: Given a web search query, retrieve relevant passages that answer the query\\n\<Query\>: \<s1\>\\n\<Document\>: \<s2\>" |
| microsoft/unixcoder-base            | Dense Vector Model             | -                                                                                                                                                                            |

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
