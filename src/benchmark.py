"""Run from repository root: python src/benchmark.py"""
import csv
from pathlib import Path
import numpy as np
from env import HighwayEnv
from trajectories import generate_trajectories
from risk import estimate_risk_for_all
from planner import select_best_trajectory


def run(seed):
    np.random.seed(seed)
    ego, others = HighwayEnv().reset()
    trajs = generate_trajectories(ego, dt=0.1, horizon=3.0)
    risks = estimate_risk_for_all(trajs, others, n_samples=100,
                                  dt=0.1, horizon=3.0,
                                  rng=np.random.default_rng(seed))
    best, scores = select_best_trajectory(trajs, risks)
    return [dict(seed=seed, manoeuvre=name, selected=int(i == best),
                 score=scores[i], **risks[i])
            for i, name in enumerate(['keep', 'brake', 'lane_left', 'lane_right'])]


def main():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [r for batch in pool.map(run, range(100)) for r in batch]
    path = Path(__file__).resolve().parents[1] / 'benchmarks' / 'benchmark_results.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ['keep', 'brake', 'lane_left', 'lane_right', 'selected']:
        group = [r for r in rows if (r['selected'] if name == 'selected' else r['manoeuvre'] == name)]
        print(name, 'selected', sum(r['selected'] for r in group))
        for key in ['collision_prob', 'avg_min_distance', 'worst_min_distance', 'score']:
            a = np.array([r[key] for r in group])
            print(key, a.mean(), a.std(ddof=1), a.min(), a.max())

if __name__ == '__main__':
    main()
