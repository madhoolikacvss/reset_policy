# Simulation integration

This directory contains the SAC training integration and its own
`v4_noelastic_unchanged` simulator. It does not need the sibling
`reset_policy_simulation` directory, its virtual environment, or its data.

The local simulator contains the required `sim_core.py`, `params.py`, and
`params.yaml` copied from that version. Physics and parameter values are
preserved; the core uses a package-relative parameter import. Other simulator
versions, rendering utilities, recorded data, and generated videos are not needed.
Training records hashes and copies parameters from this local simulator.

The cable starts with slack; winding takes up slack and a taut cable transmits
the available motor pull. There is no elastic extension or spring energy.
The `string_stiffness` parameter is retained for schema compatibility but unused.

From this directory, with Python 3.10+ (for example the `reset` conda environment):

```powershell
python -m pip install -r requirements.txt
python -B -m unittest discover -s . -p 'test_*.py' -v
python -B main_sac.py --episodes 50
```

Runs are saved under `runs/`. The legacy `original_hashes.json` file is a
historical inventory only; no runtime code reads it.
