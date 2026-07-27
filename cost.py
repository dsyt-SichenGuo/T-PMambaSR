# Parameter and FLOPs calculation follows the HiT-SR work: https://github.com/XiangZ-0/HiT-SR/issues/13
import time
import torch
from os import path as osp
from fvcore.nn import FlopCountAnalysis
from basicsr.utils.options import parse_options
from basicsr.utils.registry import ARCH_REGISTRY


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


if __name__ == '__main__':
    # ====== Basic settings ======
    # upscale = 4  # Upscaling factor
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    opt, args = parse_options(root_path, is_train=True)
    upscale = opt['scale']
    print(f"scale={upscale}")
    opt = opt['network_g']
    opt['root_path'] = root_path
    network_type = opt.pop('type')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print('Using device:', device)

    # ====== Build model ======
    net = ARCH_REGISTRY.get(network_type)(**opt)
    net.to(device).eval()

    # ====== Input sample ======
    inp = torch.rand(1, 3, 720 // upscale, 1280 // upscale, device=device)
    with torch.inference_mode():
        _sync_if_needed(device)
        out = net(inp)
        _sync_if_needed(device)

    # ====== Calculate FLOPs and parameter count ======
    with torch.no_grad():
        flops = FlopCountAnalysis(net, inp)
    print("FLOPs: %.2f G" % (flops.total() / 1e9))
    total = sum(p.numel() for p in net.parameters())
    print('Number of params: %.2fK' % (total / 1e3))

    # ====== Inference speed benchmark ======
    WARMUP = 50   # Number of warmup runs
    RUNS = 100    # Number of measured runs
    times_ms = []

    with torch.inference_mode():
        # Warmup phase
        for _ in range(WARMUP):
            _sync_if_needed(device)
            _ = net(inp)
        _sync_if_needed(device)

        # Timed benchmark phase
        for _ in range(RUNS):
            _sync_if_needed(device)
            t0 = time.perf_counter()
            _ = net(inp)
            _sync_if_needed(device)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    # ====== Summarize inference speed ======
    avg_ms = sum(times_ms) / len(times_ms)
    p50_ms = sorted(times_ms)[int(0.5 * (RUNS - 1))]
    p90_ms = sorted(times_ms)[int(0.9 * (RUNS - 1))]
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    # ====== Print results ======
    print(f"\n=== Inference Benchmark ===")
    print(f"Average latency: {avg_ms:.2f} ms per image")
    print(f"p50: {p50_ms:.2f} ms | p90: {p90_ms:.2f} ms | "
          f"min/max: {min_ms:.2f}/{max_ms:.2f} ms")
    print(f"Input shape:  {tuple(inp.shape)}  ->  Output shape: {tuple(out.shape)}")
