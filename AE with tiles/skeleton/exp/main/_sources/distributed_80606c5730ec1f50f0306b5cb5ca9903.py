import os
import sys
import time
import subprocess
import signal
import threading
import inspect
from functools import update_wrapper
import socket
from socket import AddressFamily
from socket import SocketKind

def find_free_port(addr):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((addr, 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

def make_init_address(addr=None, port=None):
    if addr is None:
        hostname = addr or socket.gethostname()
        ips = socket.getaddrinfo(
            hostname, 0,
            family=AddressFamily.AF_INET,
            type=SocketKind.SOCK_STREAM
        )
        ips = [i[4][0] for i in ips]
        ips = [i for i in ips if i not in (None, '', 'localhost', '127.0.0.1')]
        if not ips:
            raise RuntimeError('no IPv4 interface found')
        addr = ips[0]
    port = port or find_free_port(addr)
    return 'tcp://%s:%d' % (addr, port)



def watch(procs, timeout):
    try:
        rc = None
        # while all processes are alive and no error occurred
        while rc is None or rc == 0:
            time.sleep(1)
            # stop if main thread has finished
            if not threading.main_thread().isAlive:
                break
            # check if any subprocess has finished
            for proc in procs:
                rc = proc.poll() or rc
    finally:
        # wait for timeout seconds for subprocesses to finish
        end = time.time() + timeout
        rc = 0
        # check state of remaining subprocesses
        while procs:
            proc = procs.pop(0)
            rc = proc.poll() or rc
            if rc is None:
                procs.append(proc)
                time.sleep(1)
            if time.time() >= end:
                break
        for proc in procs:
            try:
                # kill remaining processes
                os.kill(proc.pid, signal.SIGKILL)
            except:
                pass
        # exit immediately if any subprocess has crashed
        if rc != 0:
            sys.exit(1)


def distribute(fn, observe=False, timeout=60):
    # update NCCL environment
    os.environ['NCCL_SOCKET_IFNAME'] = os.environ.get(
        'NCCL_SOCKET_IFNAME', 'eth,mlx5,bond,enp1s0f'
    )
    os.environ['NCCL_IB_HCA'] = os.environ.get(
        'NCCL_IB_HCA', 'mlx5'
    )

    def wrapper(_run, _config, rank, ranks, first_rank):
        init_method = _config.get('init_method') or make_init_address()
        rank_config = dict(_config)
        rank_config['init_method'] = init_method
        rank_config.pop('rank')
        if first_rank is None:
            first_rank = rank
        rank_config['first_rank'] = first_rank
        if rank_config['world_size'] is None:
            rank_config['world_size'] = ranks

        devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if devices:
            devices = devices.split(',')
        else:
            devices = list(map(str, range(ranks)))

        if rank == first_rank and ranks > 1:
            mainfile = _run.experiment_info['mainfile']
            cmd1 = [sys.executable, mainfile, 'with']
            if not observe:
                cmd1.insert(2, '-u')
            cmd2 = ['%s=%r' % (k, v) for k, v in rank_config.items()]
            procs = []
            for r in range(1, ranks):
                procs.append(subprocess.Popen(
                    cmd1 + ['rank=%d' % (first_rank+r)] + cmd2,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    encoding='utf-8',
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[r]),
                    cwd=_run.experiment_info['base_dir']
                ))
            t = threading.Thread(target=watch, args=(procs, timeout), name='watchdog')
            t.start()

        rank_config['_run'] = _run
        rank_config['_config'] = _config
        rank_config['rank'] = rank
        rank_config['ranks'] = ranks
        # remove args that fn doesn't accept
        fn_args = inspect.getfullargspec(fn).args
        for name in list(rank_config.keys()):
            if name not in fn_args:
                rank_config.pop(name, None)

        os.environ['CUDA_VISIBLE_DEVICES'] = devices[0]
        # bind parameters and call fn
        bound = inspect.signature(fn).bind(**rank_config)
        fn(*bound.args, **bound.kwargs)

    # pretend the wrapper is the wrapped
    update_wrapper(wrapper, fn)
    # but tell sacred to use our signature
    delattr(wrapper, '__wrapped__')
    return wrapper
