"""
GraphBench main runner.

Modes:
  python main.py [time_limit] [start_idx]
      Full benchmark from start_idx onward.

  python main.py [time_limit] --retest
      Read results/results.json, retest only the non-OK conjectures,
      and merge the new results back into the file.

Score = elapsed seconds per found conjecture (5s found = 5 pts).
        Timeout/fail = 120 pts penalty.
"""
import sys, json, os, time
import concurrent.futures
from conjecture import load_benchmark, to_graph6
from solver import search

_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_PATH = os.path.join(_DIR, '..', 'benchmark', 'benchmark.xlsx')
RESULTS_DIR = os.path.join(_DIR, '..', 'results')
FAIL_PENALTY = 120


def run_one(conj, time_limit):
    hard_limit = int(time_limit * 1.3) + 5
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(search, conj, time_limit)
    try:
        graph, inv, score, elapsed = future.result(timeout=hard_limit)
        status = "OK" if score > 1e-9 else "FAIL"
        return graph, inv, score, elapsed, status
    except concurrent.futures.TimeoutError:
        return None, {}, float('-inf'), float(hard_limit), "TO"
    except Exception as e:
        print(f"  ERR conj {conj.id}: {type(e).__name__}: {e}")
        return None, {}, float('-inf'), 0.0, "ERR"
    finally:
        ex.shutdown(wait=False)


def _save_json(out_path, results, found, total_done, total_conj, total_score):
    output = {
        'total_score': round(total_score, 2),
        'found': found,
        'total_done': total_done,
        'total': total_conj,
        'results': results,
    }
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, out_path)


def _make_result(conj, graph, score, elapsed, status):
    return {
        'conjecture_id': conj.id,
        'status': status,
        'g6': to_graph6(graph) if status == "OK" and graph else "",
        'violation': float(score) if score not in (float('-inf'), None) else -999,
        'time': round(elapsed, 3),
        'score_pts': round(elapsed, 3) if status == "OK" else FAIL_PENALTY,
        'x_name': conj.x_name,
        'y_name': conj.y_name,
        'subgroup': conj.subgroup,
        'n_nodes': graph.number_of_nodes() if graph else 0,
    }


def _cls(conj):
    return 'T' if 'tree' in conj.subgroup else ('C' if 'claw_free' in conj.subgroup else 'G')


def run_benchmark(time_limit=60, verbose=True, start_idx=0):
    conjectures = load_benchmark(BENCHMARK_PATH)
    n = len(conjectures)
    print(f"=== GraphBench — Full Run ===")
    print(f"Total: {n} | Time limit: {time_limit}s | Start: {start_idx}")
    print(f"Score: elapsed s per conjecture (fail/TO = {FAIL_PENALTY}s penalty)\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'results.json')

    results = []
    found = 0
    total_score = 0.0
    wall_start = time.time()

    for i, conj in enumerate(conjectures):
        if i < start_idx:
            continue

        graph, _, score, elapsed, status = run_one(conj, time_limit)
        result = _make_result(conj, graph, score, elapsed, status)

        total_score += result['score_pts']
        if status == "OK":
            found += 1
        results.append(result)
        _save_json(out_path, results, found, i + 1, n, total_score)

        if verbose:
            v_str = f"{score:.4f}" if score not in (float('-inf'), None) else "  -inf"
            wall = time.time() - wall_start
            print(
                f"[{i+1:3d}/{n}] {conj.id:5d} {_cls(conj)} {status:4s} "
                f"v={v_str:>10}  t={elapsed:6.2f}s  "
                f"pts={result['score_pts']:6.1f}  total={total_score:8.1f}  "
                f"wall={wall:5.0f}s | {conj.x_name} vs {conj.y_name}"
            )

    _print_summary(results, total_score, time.time() - wall_start, n)
    print(f"Results saved to {out_path}")
    return results, total_score


def run_retest(time_limit=60, verbose=True):
    """
    Read results.json, find every non-OK conjecture, retest it,
    and write the merged results back to results.json.
    """
    out_path = os.path.join(RESULTS_DIR, 'results.json')
    if not os.path.exists(out_path):
        print("No results.json found — run a full benchmark first.")
        return

    with open(out_path) as f:
        data = json.load(f)

    all_results = data['results']
    conjectures = load_benchmark(BENCHMARK_PATH)
    conj_by_id = {c.id: c for c in conjectures}
    n = data['total']

    # Split: keep the OK ones, retest the rest
    ok_results = {r['conjecture_id']: r for r in all_results if r['status'] == 'OK'}
    fail_ids = [r['conjecture_id'] for r in all_results if r['status'] != 'OK']
    # Also include conjectures never attempted
    done_ids = {r['conjecture_id'] for r in all_results}
    never_done = [c.id for c in conjectures if c.id not in done_ids]

    to_test = [conj_by_id[cid] for cid in fail_ids + never_done if cid in conj_by_id]

    print(f"=== GraphBench — Retest Failures ===")
    print(f"Keeping {len(ok_results)} OK results unchanged.")
    print(f"Retesting {len(to_test)} conjectures (failures + not-yet-done) | Time limit: {time_limit}s\n")

    wall_start = time.time()
    new_results = {}   # id -> result
    improved = 0

    for k, conj in enumerate(to_test):
        graph, _, score, elapsed, status = run_one(conj, time_limit)
        result = _make_result(conj, graph, score, elapsed, status)
        new_results[conj.id] = result

        old_status = 'new' if conj.id in never_done else \
                     next(r['status'] for r in all_results if r['conjecture_id'] == conj.id)
        arrow = ''
        if old_status != 'OK' and status == 'OK':
            arrow = '  ← FIXED'
            improved += 1

        if verbose:
            v_str = f"{score:.4f}" if score not in (float('-inf'), None) else "  -inf"
            wall = time.time() - wall_start
            print(
                f"[{k+1:3d}/{len(to_test)}] {conj.id:5d} {_cls(conj)} "
                f"{old_status:4s}→{status:4s} "
                f"v={v_str:>10}  t={elapsed:6.2f}s  "
                f"wall={wall:5.0f}s | {conj.x_name} vs {conj.y_name}{arrow}"
            )

    # Merge: OK results keep their original score; new results replace failures
    merged = list(ok_results.values())
    for r in new_results.values():
        merged.append(r)

    # Restore original benchmark order
    order = {c.id: i for i, c in enumerate(conjectures)}
    merged.sort(key=lambda r: order.get(r['conjecture_id'], 9999))

    found = sum(1 for r in merged if r['status'] == 'OK')
    total_score = sum(r['score_pts'] for r in merged)

    _save_json(out_path, merged, found, len(merged), n, total_score)
    _print_summary(merged, total_score, time.time() - wall_start, n)
    print(f"\nFixed {improved}/{len(to_test)} previously-failing conjectures.")
    print(f"Results saved to {out_path}")
    return merged, total_score


def _print_summary(results, total_score, wall_total, n):
    found = sum(1 for r in results if r['status'] == 'OK')
    fails = [r for r in results if r['status'] != 'OK']
    print(f"\n{'='*60}")
    print(f"Found    : {found}/{len(results)}  (benchmark total: {n})")
    print(f"Score    : {total_score:.1f}  (lower is better — 1s = 1 pt)")
    print(f"Avg/conj : {total_score / max(len(results), 1):.2f} pts")
    print(f"Wall time: {wall_total:.1f}s")
    if fails:
        print(f"\nNot found ({len(fails)}):")
        for r in fails:
            print(f"  {r['conjecture_id']:5d} [{r['status']:4s}] "
                  f"{r['x_name']} vs {r['y_name']}  ({r['subgroup']})")


if __name__ == '__main__':
    tl = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    if len(sys.argv) > 2 and sys.argv[2] == '--retest':
        run_retest(time_limit=tl)
    else:
        start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        run_benchmark(time_limit=tl, verbose=True, start_idx=start_idx)
