# FAQ

## 1 Container startup failed with error: Available device not found

Possible causes:

- When installing the driver on the host machine, the `--install-for-all` parameter was not configured, and the container runs as a regular user (HwHiAiUser), resulting in the container being unable to access the NPU card.
- The specified card is already in use by another container

Solution approach：

- Execute the following command to locate

```bash
# Start container
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
    --entrypoint bash \
    -it swr.cn-south-1.myhuaweicloud.com/ascendhub/mis-tei:<tag>

# Enter the container
docker exec -it tei_container bash

# Check if NPU info is accessible
npu-smi info
```

## 2 Container startup failed with error: "Error: Could not create backend"

Run and enter the container by referring to [FAQ 1](#1-container-startup-failed-with-error-available-device-not-found).
Inside the container, run:

```bash
# Switch to backend directory
cd /home/HwHiAiUser/text-embeddings-inference/backends/python/server

# Launch the backend service to check for stack traces
python3 text_embeddings_server/cli.py $MODEL_DIR/bge-large-zh-v1.5 (modify to your actual model path)

# If logs show "Server started at unix://tmp/text-embeddings-server", the backend has started successfully
```

## 3 NPU memory keeps growing under high concurrency or exists OOM problem

- **Solution 1 (Recommended)** : Set a small value for the environment variable `-e MAX_CONCURRENT_REQUESTS` at service startup (e.g., 64), to limit the maximum concurrent requests.See reference: [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)
- **Solution 2** : Add the environment variable `-e CLEAN_NPU_CACHE=True` at container startup to clear NPU cache after each inference.Note: This may reduce inference performance.
- **Solution 3** : Limit the maximum tokens per batch by setting `-e MAX_BATCH_TOKENS` at startup.See reference: [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)

## 4 All containers run inference only on the first NPU card when starting multiple containers

Possible reason: after adding the privileged option `--privileged=true` when the container starts, all containers run on the first card.
Solution approach:

- Only mount the specified card when the container starts, for example `--device=/dev/davinci1`
- Add environment variable to the container startup to limit running cards, for example `-e ASCEND_RT_VISIBLE_DEVICES=1`

## 5 Run multiple containers/services on a single NPU card

**Solution 1** : Start the container with:
`-u root --privileged=true -e ASCEND_RT_VISIBLE_DEVICES=<target_card_id>`
Note: Privileged mode grants access to all NPUs by default.

**Solution 2** : Run and enter the container by referring to [FAQ 1](#1-container-startup-failed-with-error-available-device-not-found).

Inside the container, start multiple services:

```bash
bash start.sh --model-id <absolute_path_to_model_1> --port <listen_port_1> &
bash start.sh --model-id <absolute_path_to_model_2> --port <listen_port_2> &
```

## 6 Container startup failed with error: "failed to create HTTP listener: Address already in use (os error 98)"

Multiple TEI containers are running on the host with host network mode.The default PROMETHEUS_PORT is 9000 and is already in use.
Fix: Assign a different port at startup,e.g.,`-e PROMETHEUS_PORT=9001`

## 7 Container startup failed with error: "`--max-batch-tokens` cannot be lower than the model `max_input_length`"

Error message:

Error: `--max-batch-tokens` cannot be lower than the model `max_input_length` (32768 < 16384) when `--auto-truncate` is disabled

Solution approach: Enable auto-truncation at container startup:
`-e AUTO_TRUNCATE=True`
See reference: [CLI arguments](https://gitcode.com/Ascend/text-embeddings-inference/blob/main/docs/source/en/cli_arguments.md)

## 8 In-container service fails to start and requires adding `--security-opt="seccomp=unconfined"`

Symptom: After starting the container with the documented command, the TEI service inside the container fails to start or the process exits immediately, with errors such as `Operation not permitted`, `Permission denied`, or no visible error before exit (e.g., system calls like `clone3` used by glibc/Python multi-threading being blocked by the default seccomp profile).

Possible causes: Docker enables a default seccomp security profile for containers that blocks some system calls. The CANN driver stack, Python/torch_npu backend that TEI depends on may invoke these blocked system calls during initialization, causing the service to fail to start. This is more likely after Docker/containerd upgrades, CANN version changes, or when migrating from `--privileged` to the `--device` mounting approach.

Solution: Add `--security-opt="seccomp=unconfined"` to the container startup command to disable the default seccomp profile:

```bash
docker run \
    -u <user> \
    --security-opt="seccomp=unconfined" \
    --device /dev/davinci0 \
    ...(rest of the parameters identical to the startup command in [Quick Start](OVERVIEW.md#running-a-tei-container))
```

Notes:

- This parameter only disables syscall filtering; it does not grant additional device access or other privileges. Keep mounting the required cards explicitly with `--device` and do not mix it with `--privileged`.
- Difference from `--privileged`: `--privileged` implicitly disables seccomp and grants the container almost all host privileges, and may cause multiple containers to run on the first NPU card (see [FAQ 4](#4-all-containers-run-inference-only-on-the-first-npu-card-when-starting-multiple-containers)); `--security-opt="seccomp=unconfined"` only disables syscall filtering and has a smaller blast radius.
