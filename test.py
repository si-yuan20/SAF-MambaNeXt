# -*- coding: utf-8 -*-
"""
Benchmark Inference Time (ms), FLOPs (G), Params (M) for:
- ConvNeXt
- Mamba
- ConvNeXt+Mamba (cat baseline)
- ConvNeXt+Mamba+SAF
- ConvNeXt+Mamba+UGBF
- ConvNeXt+Mamba+UGBF+SAF

Notes:
1) Uses existing modules in dual_model.py/config.py without changing original model code.
2) FLOPs requires 'thop' (pip install thop). If missing, will warn and continue.
3) Mamba-SSM usually requires CUDA; CPU may fail for mamba branch.
"""

import os
import time
import argparse
import torch
import torch.nn as nn

from config import make_default_cfg, AblationConfig  # from your project
from dual_model import ConvNeXtNet, MambaNet, DualConvNeXtMambaNet  # from your project

def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def measure_infer_time_ms(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int = 30,
    iters: int = 200,
) -> float:
    """
    Accurate GPU timing: synchronize before/after.
    Returns mean latency in ms per forward.
    """
    model.eval()

    # warmup
    for _ in range(max(0, warmup)):
        _ = model(x)

    if x.is_cuda:
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(max(1, iters)):
        _ = model(x)

    if x.is_cuda:
        torch.cuda.synchronize()

    t1 = time.time()
    avg_ms = (t1 - t0) * 1000.0 / max(1, iters)
    return avg_ms


def measure_flops_g(model: nn.Module, x: torch.Tensor) -> float | None:
    """
    FLOPs via thop.profile (MACs -> FLOPs approx).
    Return FLOPs in G, or None if thop not available.
    """
    try:
        from thop import profile  # type: ignore
    except Exception:
        return None

    model.eval()
    # thop expects tuple input
    macs, params = profile(model, inputs=(x,), verbose=False)
    # Common convention: FLOPs ≈ 2 * MACs
    flops = 2.0 * float(macs)
    return flops / 1e9

def load_convnext_local_weights(backbone: nn.Module, ckpt_path: str, strict: bool = False) -> None:
    """
    backbone: ConvNeXtNet.backbone DualConvNeXtMambaNet.convnext_backbone
    """
    if not ckpt_path or (not os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"[ConvNeXt] local weight not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    msg = backbone.load_state_dict(state, strict=strict)

    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    print(f"[ConvNeXt] Loaded LOCAL weights from: {ckpt_path}")
    print(f"[ConvNeXt] Missing={len(missing)}, Unexpected={len(unexpected)}")

def try_load_convnext_pretrained(model: nn.Module, pretrained_path: str):
    """
    For ConvNeXtNet: model.backbone is timm features_only or torchvision features.
    Your training pipeline loads convnext weights into the backbone. Here we do best-effort load.
    If fails, we continue (benchmarking still valid for time/params; FLOPs unaffected).
    """
    if not pretrained_path or (not os.path.isfile(pretrained_path)):
        print(f"[WARN] pretrained_path not found, skip loading: {pretrained_path}")
        return

    try:
        state = torch.load(pretrained_path, map_location="cpu")
        # ConvNeXtNet uses self.backbone
        msg = model.backbone.load_state_dict(state, strict=False)
        print(f"[ConvNeXt] Loaded pretrained from: {pretrained_path}")
        print(f"[ConvNeXt] Missing={len(getattr(msg,'missing_keys',[]))}, Unexpected={len(getattr(msg,'unexpected_keys',[]))}")
    except Exception as e:
        print(f"[WARN] Failed to load convnext pretrained weights: {e}")


def build_models(num_classes: int, img_size: int, cfg) -> list[tuple[str, nn.Module]]:
    """
    Create the 6 requested methods.
    """
    models: list[tuple[str, nn.Module]] = []

    # 1) ConvNeXt
    m1 = ConvNeXtNet(num_classes=num_classes, pretrained=False)
    models.append(("ConvNeXt", m1))

    # 2) Mamba
    m2 = MambaNet(num_classes=num_classes, img_size=img_size)
    models.append(("Mamba", m2))

    # helper to make ablation config
    def ab(use_saf: bool, saf_prior: str, use_ugbf: bool) -> AblationConfig:
        return AblationConfig(
            use_convnext=True,
            use_mamba=True,
            use_saf=use_saf,
            saf_prior=saf_prior,
            saf_dim=cfg.model.dual.saf_dim,
            saf_fuse=cfg.model.dual.saf_fuse,
            use_ugbf=use_ugbf,
            ugbf_temperature=cfg.model.dual.ugbf_temperature,
            detach_gate=cfg.model.dual.detach_gate,
            gate_min=cfg.model.dual.gate_min,
        )

    # 3) ConvNeXt+Mamba（baseline cat）=> SAF off, UGBF off
    m3 = DualConvNeXtMambaNet(
        num_classes=num_classes,
        convnext_pretrained=False,
        mamba_img_size=img_size,
        ablation=ab(use_saf=False, saf_prior="none", use_ugbf=False),
    )
    models.append(("ConvNeXt+Mamba", m3))

    # 4) ConvNeXt+Mamba+SAF
    m4 = DualConvNeXtMambaNet(
        num_classes=num_classes,
        convnext_pretrained=False,
        mamba_img_size=img_size,
        ablation=ab(use_saf=True, saf_prior="edge", use_ugbf=False),
    )
    models.append(("ConvNeXt+Mamba+SAF", m4))

    # 5) ConvNeXt+Mamba+UGBF
    m5 = DualConvNeXtMambaNet(
        num_classes=num_classes,
        convnext_pretrained=False,
        mamba_img_size=img_size,
        ablation=ab(use_saf=False, saf_prior="none", use_ugbf=True),
    )
    models.append(("ConvNeXt+Mamba+UGBF", m5))

    # 6) ConvNeXt+Mamba+UGBF+SAF
    m6 = DualConvNeXtMambaNet(
        num_classes=num_classes,
        convnext_pretrained=False,
        mamba_img_size=img_size,
        ablation=ab(use_saf=True, saf_prior="edge", use_ugbf=True),
    )
    models.append(("ConvNeXt+Mamba+UGBF+SAF", m6))

    return models


def print_table(rows: list[dict]):
    # pretty print
    headers = ["Method", "Inference time(ms)", "FLOPs(G)", "Parameters(M)"]
    colw = [28, 18, 10, 14]
    line = "=" * (sum(colw) + 3 * (len(colw) - 1))
    print("\n" + line)
    print(f"{headers[0]:<{colw[0]}} | {headers[1]:<{colw[1]}} | {headers[2]:<{colw[2]}} | {headers[3]:<{colw[3]}}")
    print(line)
    for r in rows:
        print(
            f"{r['Method']:<{colw[0]}} | "
            f"{r['Inference time(ms)']:<{colw[1]}} | "
            f"{r['FLOPs(G)']:<{colw[2]}} | "
            f"{r['Parameters(M)']:<{colw[3]}}"
        )
    print(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=None, help="input image size, default from config")
    parser.add_argument("--batch_size", type=int, default=1, help="benchmark batch size (recommend 1 for latency)")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument(
        "--convnext_pretrained_path",
        type=str,
        default=None,
        help="LOCAL convnext pth path. If not set, will use cfg.model.pretrained_path",
    )
    args = parser.parse_args()

    cfg = make_default_cfg()
    img_size = args.img_size if args.img_size is not None else cfg.data.img_size

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA not available, fallback to CPU. Mamba may fail on CPU.")

    # dummy input
    x = torch.randn(args.batch_size, 3, img_size, img_size, device=device)

    local_ckpt = args.convnext_pretrained_path or getattr(cfg.model, "pretrained_path", "")
    if not local_ckpt:
        print("[WARN] convnext local weight path is empty. ConvNeXt branch will run with random init.")

    models = build_models(num_classes=args.num_classes, img_size=img_size, cfg=cfg)

    rows = []
    for name, model in models:
        model = model.to(device)

        # Dual: DualConvNeXtMambaNet.convnext_backbone
        if local_ckpt:
            try:
                if name == "ConvNeXt" and hasattr(model, "backbone"):
                    load_convnext_local_weights(model.backbone, local_ckpt, strict=False)
                elif "ConvNeXt+Mamba" in name and hasattr(model, "convnext_backbone"):
                    load_convnext_local_weights(model.convnext_backbone, local_ckpt, strict=False)
            except Exception as e:
                print(f"[WARN] Local ConvNeXt weight load failed for {name}: {e}")

        # some mamba ops require cuda; if cpu, skip time/flops with clear note
        if ("Mamba" in name) and device.type != "cuda":
            rows.append({
                "Method": name,
                "Inference time(ms)": "SKIP(cpu)",
                "FLOPs(G)": "SKIP(cpu)",
                "Parameters(M)": f"{count_params_m(model):.6f}",
            })
            continue

        # Inference time
        try:
            t_ms = measure_infer_time_ms(model, x, warmup=args.warmup, iters=args.iters)
            t_ms_str = f"{t_ms:.4f}"
        except Exception as e:
            t_ms_str = f"ERR({type(e).__name__})"

        # FLOPs
        flops_g = None
        try:
            flops_g = measure_flops_g(model, x)
        except Exception:
            flops_g = None

        flops_str = f"{flops_g:.3f}" if flops_g is not None else "N/A(thop)"

        # Params
        params_m = count_params_m(model)

        rows.append({
            "Method": name,
            "Inference time(ms)": t_ms_str,
            "FLOPs(G)": flops_str,
            "Parameters(M)": f"{params_m:.6f}",
        })

    print_table(rows)

    # also save csv
    out_csv = "benchmark_results.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("Method,Inference time(ms),FLOPs(G),Parameters(M)\n")
        for r in rows:
            f.write(f"{r['Method']},{r['Inference time(ms)']},{r['FLOPs(G)']},{r['Parameters(M)']}\n")
    print(f"[Saved] {out_csv}")



if __name__ == "__main__":
    main()
