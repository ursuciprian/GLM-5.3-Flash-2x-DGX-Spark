# Node guard

Prevents the class of outage that cost two power cycles on 2026-08-27.

## The problem

GB10 shares one 128 GB pool between host and GPU. A server that over-allocates
starves userspace; sshd is among the first casualties, so the box stays
pingable with port 22 open while never completing a banner exchange. At that
point there is no remote way in and only a power cycle recovers it.

Two things that did NOT prevent it:

- **`gpu-memory-utilization` / `mem-fraction-static`.** These bound weights and
  KV. They do not bound torch.compile workspaces, CUDA graph capture buffers,
  JIT/autotune warmup, or NCCL buffers. vLLM at 0.70 held 42 GB free through
  the entire weight load and then locked up during graph capture.
- **earlyoom as configured.** `-m 2` acts at ~2.5 GB free on a 128 GB box.
  sshd fails to accept new connections far above that, so there is a wide
  unmanageable band that earlyoom sleeps through.

- **A watchdog over SSH.** Dies with the thing it is watching. It has to run on
  the node.

## Install (both nodes)

```sh
sudo cp spark-memguard.sh /usr/local/bin/
sudo cp spark-memguard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now spark-memguard
journalctl -u spark-memguard -f      # or tail /var/log/spark-memguard.log
```

Defaults: warn below 25 GB, stop inference below 15 GB, poll every 5 s.
Override with `MEMGUARD_THRESHOLD_GB`, `MEMGUARD_GRACE_GB`, `MEMGUARD_INTERVAL`.

15 GB is chosen to sit above where sshd starts failing and below any healthy
steady state measured here (SGLang balanced idles at 16-17 GB free, which is
close — raise the threshold only after checking your config's steady state, or
it will stop a healthy server).

## Also worth doing

Raise earlyoom's threshold so the backstop is not purely decorative:

```sh
# /etc/default/earlyoom
EARLYOOM_ARGS="-m 10 -s 80 --prefer '(vllm|VLLM|sglang|llama-server|python3)' --avoid '(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager)'"
sudo systemctl restart earlyoom
```

And bound the workload's untracked transients at launch:

- vLLM: `--enforce-eager` (or `-O0`) to skip compile and graph capture
- SGLang: `--disable-flashinfer-autotune`, `--disable-prefill-cuda-graph`,
  capped `--cuda-graph-max-bs-decode`

The strongest fix, not yet available: a hard cgroup limit on the container so
the kernel refuses the allocation and the server OOMs instead of the host.
sparkrun's `executor_config` whitelist does not expose `--memory`; it would
need that feature or a `systemd-run --scope -p MemoryMax=100G` wrapper.
