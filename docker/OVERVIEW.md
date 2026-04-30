# Ascend text-embeddings-inference(TEI)

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- Where to get help

  - [Issue Feedback](https://gitcode.com/Ascend/text-embeddings-inference/issues)
  - [text-embeddings-inference Code](https://gitcode.com/Ascend/text-embeddings-inference)
  - [text-embeddings-inference API Reference](../docs/zh/api/README.md)
  - [AscendHub Image Repository](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d)
  - [Ascend Community](https://www.hiascend.com/)

## Supported Tags and Dockerfile Links

### Tag Naming Convention

Tags follow this pattern:

```bash
<TEI_version>-<chip_series>-<os>-<python_version>
```

| Field            | Example Values                  | Description                       |
|------------------|---------------------------------|-----------------------------------|
| `TEI_version`    | `26.0.0`                        | text-embeddings-inference version |
| `chip_series`    | `910b`, `a3`, `310p`            | Target Ascend chip family         |
| `os`             | `ubuntu22.04`, `openeuler24.03` | Base operating system             |
| `python_version` | `py3.11`                        | Python version                    |

### CANN 9.0.0 + 26.0.0 TEI Image

| Tag                                 | Dockerfile                                | Image Content |
|-------------------------------------|-------------------------------------------|---------------|
| `26.0.0-910b-ubuntu22.04-py3.11`    | [Dockerfile](./Dockerfile.910b.ubuntu)    | cann + TEI    |
| `26.0.0-910b-openeuler24.03-py3.11` | [Dockerfile](./Dockerfile.910b.openeuler) | cann + TEI    |
| `26.0.0-a3-ubuntu22.04-py3.11`      | [Dockerfile](./Dockerfile.a3.ubuntu)      | cann + TEI    |
| `26.0.0-a3-openeuler24.03-py3.11`   | [Dockerfile](./Dockerfile.a3.openeuler)   | cann + TEI    |
| `26.0.0-310p-ubuntu22.04-py3.11`    | [Dockerfile](./Dockerfile.310p.ubuntu)    | cann + TEI    |
| `26.0.0-310p-openeuler24.03-py3.11` | [Dockerfile](./Dockerfile.310p.openeuler) | cann + TEI    |


---

## Quick Start

### Prerequisites (optional)

#### Install Driver

An Ascend NPU driver compatible with the container's CANN version must be installed on the host. See the [CANN Compatibility Matrix](https://www.hiascend.com/document) for driver ↔ CANN version mapping.

---

### Running a TEI Container

```bash
# Configure the host machine model storage directory
export MODEL_DIR="/home/HwHiAiUser/model"
# Replace <tag> with the actual tag, <model name> with the actual model name, and <ip> with the actual IP
docker run \
    --name tei_container \
    --device /dev/davinci1 \
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
    -p 8000:8000 \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag> --model-id $MODEL_DIR/<model name> --hostname <ip> --port 8000
```

### How to build docker image

```bash
# Clone the code in any directory on the host machine, then go into the docker directory, and get the Dockerfile. Replace {cann version} with the actual version, and replace {your_repo} with the actual repository.
git clone https://gitcode.com/Ascend/text-embeddings-inference.git && cd text-embeddings-inference/docker

docker build --build-arg CANN_VERSION={cann version} -t {your_repo}/mis-tei:latest -f Dockerfile.<chip_series>.<os> .
```

### Development


---

## Supported Hardware

| Chip Series | Product Examples               | Architecture   |
|-------------|--------------------------------|----------------|
| Ascend 910B | Atlas 800I A2                  | ARM64 / x86_64 |
| Ascend A3   | Atlas 800I A3                  | ARM64 / x86_64 |
| Ascend 310P | Atlas 300I Pro, Atlas 300V Pro | ARM64 / x86_64 |

---

## License

View the [license information](https://github.com/Ascend/cann-container-image/blob/main/LICENSE) for CANN and MindSeries software included in these images.

As with all container images, the pre-installed packages (Python, system libraries, etc.) may be subject to their own licenses.
