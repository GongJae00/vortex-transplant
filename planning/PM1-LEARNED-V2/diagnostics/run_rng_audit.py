"""Lightweight PlainConvRNN RNG variability audit.
Checks whether model state differs across processes by hashing parameters.
"""
import hashlib, json, multiprocessing, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

N_RUNS = 10

def params_fingerprint(model):
    """SHA-256 of all parameter tensors."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run_one(args):
    run_id, = args
    import torch
    dev = torch.device("cpu")
    from topological.model import ModelSpec
    from topological.training import make_model
    spec = ModelSpec()
    result = {}
    for mt in ["u1", "plain"]:
        model = make_model(0, model_type=mt, model_spec=spec, device=dev)
        result[f"{mt}_params_sha256"] = params_fingerprint(model)
        result[f"{mt}_seed_after"] = torch.initial_seed()
        del model
    return result

if __name__ == "__main__":
    t0 = time.perf_counter()
    with multiprocessing.Pool(min(N_RUNS, multiprocessing.cpu_count())) as pool:
        runs = pool.map(run_one, [(i,) for i in range(N_RUNS)])
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.1f}s ({N_RUNS} runs)")

    for mt in ["u1", "plain"]:
        key = f"{mt}_params_sha256"
        fps = [r[key] for r in runs]
        n_unique = len(set(fps))
        print(f"{mt}: {n_unique} unique fingerprints across {N_RUNS} processes")
        if n_unique > 1:
            print(f"  *** {mt} parameter state is NOT deterministic across processes ***")
            for i, fp in enumerate(fps):
                print(f"    run {i}: {fp[:16]}...")
        else:
            print(f"  ✓ deterministic: {fps[0][:16]}...")

    out_path = os.path.join(os.path.dirname(__file__), "raw", "rng_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"elapsed_s": elapsed, "n_runs": N_RUNS, "results": [{f"{mt}_{k}": v for mt in ["u1","plain"] for k,v in r.items()} for r in runs]}, f, indent=2)
    print(f"Saved to {out_path}")
