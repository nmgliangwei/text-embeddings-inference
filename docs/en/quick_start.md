# Quick Start

This tutorial guides you through deploying the `BAAI/bge-large-zh-v1.5` model using the TEI image, enabling you to quickly master TEI deployment and usage.

## Step 1: Pulling the Image

1. **Determining the image version to pull**
   - Visit the Ascend community [image repository](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d) to obtain the latest image version and the matching CANN version.
   - Select the matching version based on the current hardware model (for example, the Atlas 800I A2 inference server).
   > [!NOTE]
   >
   > - CANN is preinstalled in the image, and you do not need to install it again.
   > - Note the difference between CPU architectures (x86_64/aarch64).

2. **Prechecking the environment**
   - Run the `npu-smi info` command to obtain the NPU driver version.
   - Use the CANN version to determine the matching [firmware and driver combination](https://www.hiascend.com/hardware/firmware-drivers/community). If the driver version does not match, refer to the [firmware and driver documentation](https://www.hiascend.com/hardware/firmware-drivers/community) to update the NPU driver.

3. **Pulling the image**
   The image tag format is `{version}-{chip}-{os}-{python}-{arch}`, and each variable is described as follows:

 | Field         | Example Value                      | Description        |
|------------|--------------------------------|-----------|
| TEI version | `26.0.0`                       | TEI version number|
| Chip series | `910`, `a3`, `Atlas 300I Pro`  | Target chip series |
| Operating system | `ubuntu22.04`, `openeuler24.03` | Base operating system |
| Python version | `py3.11`                       | Python version |

First, ensure that NPU devices are present in your environment and that the NPU driver and firmware are installed. Then pull the [TEI image](https://www.hiascend.com/developer/ascendhub/detail/07a016975cc341f3a5ae131f2b52399d) version that matches the current hardware. This example uses version 26.0.0 with the 910b chip, Ubuntu 22.04, and Python 3.11.
```bash
    docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11
```
## Step 2: Downloading the Model Weights
Download the model weights to the `/home/data` directory. This example uses the `bge-large-zh-v1.5` model.

```bash
   cd /home/data
   GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/BAAI/bge-large-zh-v1.5.git
   cd bge-large-zh-v1.5 && git lfs pull
 ```
## Step 3: Running the Container

```bash
    # Set the directory for storing the model weights
    export MODEL_DIR=/home/data
    docker run -itd \
    -u root \
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
    -itd swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:26.0.0-910b-ubuntu22.04-py3.11 --model-id $MODEL_DIR/bge-large-zh-v1.5 --hostname 127.0.0.1 --port 8000
  ```

   - You can run the `ls -al /dev/davinci*` command to check which device files can be mounted.
   - For the parameters of the container startup command, refer to [CLI arguments](https://atomgit.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md).

## Step 4: Testing Service Requests
Access the service to obtain the vector results.
   ```bash
   curl 127.0.0.1:8080/v1/embeddings \
       -X POST \
       -d '{"input":["The capital of China is Beijing."]}' \
       -H 'Content-Type: application/json'
   ```
