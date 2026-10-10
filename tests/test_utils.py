import subprocess
import sys
import time
import signal
import unittest
import requests
import numpy as np
import os


class TestBase(unittest.TestCase):
    """测试基类，提供公共的设置和清理方法"""
    model_id = None
    port = None
    env = None
    base_url = None
    process = None

    @classmethod
    def setUpClass(cls):
        model_dir=os.environ.get("MODEL_DIR", "/home/data")
        cls.process = _start_service(os.path.join(model_dir, cls.model_id), cls.port, env=cls.env)

    @classmethod
    def tearDownClass(cls):
        _stop_service(cls.process)


def _compute_cosine_distance(emb1, emb2):
    """计算两个嵌入向量的余弦距离"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def _wait_for_health(port, timeout=180, interval=5):
    """等待服务健康检查通过"""
    base_url = f"http://127.0.0.1:{port}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def _start_service(model_id, port, env=None):
    """启动服务"""
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    print(f"Starting service for model {model_id}")
    process = subprocess.Popen(
        [
            "/bin/bash",
            "start.sh",
            "--model-id",
            model_id,
            "--port",
            str(port),
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=proc_env,
        shell=False,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    if not _wait_for_health(port):
        _stop_service(process)
        raise RuntimeError(
            f"Service for model '{model_id}' failed to start within the timeout"
        )
    return process


def _collect_descendants(pid):
    """通过 /proc 递归收集 pid 的所有子孙进程 id（Linux，不依赖 psutil）"""
    def children_of(ppid):
        result = []
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/stat", "r") as f:
                        stat = f.read()
                except OSError:
                    continue
                # stat 格式: "pid (comm) state ppid ..."，comm 可能含空格/括号，从最后一个 ')' 之后解析
                rp = stat.rfind(")")
                if rp == -1:
                    continue
                fields = stat[rp + 1:].split()
                if len(fields) >= 2 and fields[1] == str(ppid):
                    result.append(int(entry))
        except OSError:
            pass
        return result

    descendants = []
    queue = [pid]
    while queue:
        current = queue.pop(0)
        for child in children_of(current):
            descendants.append(child)
            queue.append(child)
    return descendants


def _kill_processes(pids):
    """先 SIGTERM 后 SIGKILL 终止一组进程"""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(1)
    for pid in reversed(pids):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _stop_service(process):
    """停止服务：先快照进程树再终止，确保子进程（router/python 后端）不残留"""
    if process and process.poll() is None:
        # 先收集进程树快照，避免 bash 退出后子进程成为孤儿、无法按树清理
        descendants = _collect_descendants(process.pid)
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        finally:
            _kill_processes(descendants)


def _make_test_class(model_id_, port_, env_=None, test_base_class=None, class_prefix="Test"):
    """创建测试类"""
    model_name = model_id_.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")
    class_name = f"{class_prefix}_{model_name}"

    base_class = test_base_class or TestBase

    class Cls(base_class):
        model_id = model_id_
        port = port_
        env = env_ or {}
        base_url = f"http://127.0.0.1:{port_}"

    Cls.__name__ = class_name
    Cls.__qualname__ = class_name
    return Cls
