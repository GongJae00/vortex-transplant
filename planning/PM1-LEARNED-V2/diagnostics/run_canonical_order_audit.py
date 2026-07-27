"""Canonical-order audit: multi-process Plain RNG variability check.
Tests whether PlainConvRNN charge maps vary across fresh processes
due to missing generator pass in conv.weight init (model.py:186).
"""
import json, multiprocessing, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

N_RUNS = 10

def run_one(args):
    run_id, gpu = args
    import torch
    if gpu:
        torch.cuda.set_device(1)  # use cuda:1
        dev = torch.device("cuda:1")
    else:
        dev = torch.device("cpu")
        
    from topological.model import ModelSpec
    from topological.training import make_model
    from topological.learned_evaluation import evaluate_seed_model, TEST_EXAMPLES
    from topological.interventions import hidden_to_complex
    from topological.topology import extract_charge
    from topological.task import generate_copy_batch, run_copy

    spec = ModelSpec()
    results = {}

    for model_type in ["u1", "plain"]:
        model = make_model(0, model_type=model_type, model_spec=spec, device=dev)
        record = evaluate_seed_model(model, 0, task_type="copy")
        batch = generate_copy_batch(0, "test/heldout-delay-64", TEST_EXAMPLES, 64, device=dev)
        trace = run_copy(model, batch.symbols, 64)
        h = trace.post_write.detach().cpu()
        charge_counts = []
        for ex in range(min(8, h.shape[0])):
            f = hidden_to_complex(h[ex])
            for ch in range(f.shape[0]):
                c = extract_charge(f[ch])
                charge_counts.append(int(c.charge.sum()))
        results[model_type] = {
            "net_charge_sum": sum(charge_counts),
            "defect_prevalence": record.get("defect_prevalence"),
            "torch_seed_after": torch.initial_seed(),
        }
        del model
    return {"run": run_id, **results}


if __name__ == "__main__":
    t0 = time.perf_counter()
    gpu = "--gpu" in sys.argv
    with multiprocessing.Pool(min(N_RUNS, multiprocessing.cpu_count())) as pool:
        runs = pool.map(run_one, [(i, gpu) for i in range(N_RUNS)])
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.1f}s ({N_RUNS} runs)")

    u1_charges = [r["u1"]["net_charge_sum"] for r in runs]
    u1_prev = [r["u1"]["defect_prevalence"] for r in runs]
    plain_charges = [r["plain"]["net_charge_sum"] for r in runs]
    plain_prev = [r["plain"]["defect_prevalence"] for r in runs]

    print(f"\nU1 net_charge_sum: {set(u1_charges)} unique={len(set(u1_charges))}")
    print(f"U1 defect_prevalence: {set(u1_prev)}")
    print(f"Plain net_charge_sum: {set(plain_charges)} unique={len(set(plain_charges))}")
    print(f"Plain defect_prevalence: {set(plain_prev)}")

    det_u1 = len(set(u1_charges)) == 1
    det_plain = len(set(plain_charges)) == 1
    print(f"\nU1 deterministic: {det_u1}")
    print(f"Plain deterministic: {det_plain}")

    out = {
        "elapsed_s": elapsed,
        "n_runs": N_RUNS,
        "u1_deterministic": det_u1,
        "plain_deterministic": det_plain,
        "u1_charge_sums": u1_charges,
        "plain_charge_sums": plain_charges,
        "u1_prevalence": u1_prev,
        "plain_prevalence": plain_prev,
    }
    out_path = os.path.join(os.path.dirname(__file__), "raw", "canonical_order_audit.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")
