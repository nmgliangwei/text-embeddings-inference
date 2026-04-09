#!/bin/bash
# Copyright © Huawei Technologies Co., Ltd. 2024. All rights reserved.

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 <model_path_or_id> [additional_args_for_text-embeddings-router]"
    echo ""
    echo "Arguments:"
    echo "  <model_path_or_id>    : Local model directory path OR ModelScope model ID"
    echo "                          e.g., /path/to/model or namespace/model-name"
    echo "  [additional_args]     : Any additional arguments to pass to text-embeddings-router"
    echo ""
    echo "Examples:"
    echo "  $0 /home/HwHiAiUser/model/BAAI/bge-base-zh-v1.5 --port 8080 --hostname 0.0.0.0 --served-model-name bge-base-zh-v1.5"
    echo "  $0 BAAI/bge-base-zh-v1.5 --port 8080 --hostname 0.0.0.0 --served-model-name bge-base-zh-v1.5"
    exit 1
fi

MODEL_INPUT=$1
shift
ADDITIONAL_ARGS="$@"

BASE_MODEL_DIR="/home/HwHiAiUser/model"

function is_local_path() {
    if [[ -d "$1" ]]; then
        return 0
    else
        return 1
    fi
}

function check_model_exists() {
    local model_dir="$1"
    if [[ ! -e "${model_dir}/config.json" ]]; then
        echo "Model '${model_dir}' does not exist or missing config.json."
        return 1
    else
        echo "Model '${model_dir}' exists."
        return 0
    fi
}

function download_model() {
    local model_id="$1"
    local target_dir="$2"
    local retry_time=1
    local max_retries=5
    echo "Downloading model '${model_id}' from ModelScope to ${target_dir} ..."

    while [[ ${retry_time} -le ${max_retries} ]]; do
        if (
            rm -rf "${target_dir}" && \
            git clone "https://www.modelscope.cn/${model_id}" "${target_dir}" && \
            cd "${target_dir}" && \
            git lfs pull
        ); then
            echo "Download successful."
            return 0
        fi

        retry_time=$((retry_time + 1))
        if [[ ${retry_time} -le ${max_retries} ]]; then
            echo "Download failed, retrying (${retry_time}/${max_retries})..."
            sleep 5
        fi
    done

    echo "Maximum retries ${max_retries} reached. Download failed."
    return 1
}

function start_tei_service() {
    local model_dir="$1"
    local additional_args="$2"

    if [[ -z ${TEI_NPU_DEVICE} ]]; then
        export TEI_NPU_DEVICE=0
        echo "Running on NPU device 0."
    fi

    local model_name=$(basename "${model_dir}")
    echo "Starting TEI service with model '${model_name}'..."
    echo "Additional arguments: ${additional_args}"

    # Construct the full command
    cmd="text-embeddings-router --model-id "${model_dir}" ${additional_args}"
    echo "Executing: ${cmd}"
    eval "${cmd}"
}

function main() {
    local model_dir_actual
    local use_local=false

    if is_local_path "${MODEL_INPUT}"; then
        model_dir_actual="${MODEL_INPUT}"
        use_local=true
        echo "Using local model path: ${model_dir_actual}"
    else
        local model_id="${MODEL_INPUT}"
        model_dir_actual="${BASE_MODEL_DIR}/$(basename "${model_id}")"
        use_local=false
        echo "Treating input as ModelScope ID. Model will be stored at: ${model_dir_actual}"
    fi

    if ! check_model_exists "${model_dir_actual}"; then
        if [[ "${use_local}" == true ]]; then
            echo "Error: Local model path provided but config.json not found at ${model_dir_actual}"
            exit 1
        else
            if ! download_model "${MODEL_INPUT}" "${model_dir_actual}"; then
                echo "Download model ${MODEL_INPUT} failed."
                exit 1
            fi
        fi
    fi

    start_tei_service "${model_dir_actual}" "${ADDITIONAL_ARGS}"
}
sleep 3
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/home/HwHiAiUser/.cargo/bin:$PATH"
main
