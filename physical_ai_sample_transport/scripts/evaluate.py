"""Repeat the baseline with independent ID permutations and aggregate results."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
seeds=[42,7,8,9]
for seed in seeds:
    with (ROOT/f'artifacts/eval_seed_{seed}.log').open('w') as f:
        p=subprocess.run([str(ROOT/'scripts/run.sh'),'scripts/demo.py','--ai','--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:raise SystemExit(f'Seed {seed} failed; see artifacts/eval_seed_{seed}.log')
results=[json.loads((ROOT/f'artifacts/result_seed_{s}.json').read_text()) for s in seeds]
summary={'seeds':seeds,'episodes':len(results),'sample_transfers':sum(len(t['results']) for t in results),'successful_transfers':sum(x['success'] for t in results for x in t['results']),'all_episodes_successful':all(t['all_success'] for t in results),'max_xy_error_m':max(x['xy_error_m'] for t in results for x in t['results']),'max_joint_speed_rad_s':max(t['max_joint_speed_rad_s'] for t in results),'max_joint_limit_violation_rad':max(t['max_joint_limit_violation_rad'] for t in results),'scope':'fixed pickup coordinates; shuffled marker IDs; ideal latch grasp; synthetic simulation'}
(ROOT/'artifacts/evaluation_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if not summary['all_episodes_successful']:raise SystemExit(1)
