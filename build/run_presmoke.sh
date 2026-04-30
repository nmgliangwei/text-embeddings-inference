#!/bin/bash
set -e

readonly CUR_DIR=$(dirname "$(readlink -f "$0")")

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

npu-smi info

pip3 install pytest pytest-cov pytest-html
rm -f $(which python-text-embeddings-server)
cd $CUR_DIR/..
cd backends/python/server && make -j$(nproc) && make install && pip3 cache purge
cd $CUR_DIR/..
python3 -m pytest tests/ -vs
