# Workspace guide

Launch with `uv run ginseng-tui` from this repository. Home provides a workflow table, the liquid wireframe study, and session readouts. The liquid study is decorative; dashboards and charts use computed results.

## Simulation and exact

Open Simulation to choose a synthetic fixture or a local JSON file, sampling method, path count, seed, horizon, and block length. `examples/tiny-history.json` shows the input format. Sobol requires a power-of-two path count. Exact enumerates the small reference fixture independently.

Completed runs show the required cash reserve, shortfall probability, cash paths, and daily cash signals. The archive reopens locally saved dashboards.

## Research

Press `Ctrl+R` or use `Ctrl+P` to find a recipe. Quick setup fills common fields; the JSON editor exposes the complete recipe. Run starts an isolated process, leaving navigation available. Cancel requests cooperative stopping for precision experiments and terminates other experiment processes.

Recipes cover precision stopping, the cash-buffer atlas, tail risk, sampler comparisons, funding policies, portfolio allocation, personal forecasts, calibration, and engine capture/replay. Optimization requires `--extra optimization`; report plots require `--extra research`. The native performance benchmark requires the full upstream repository.

The selected results table can be plotted using numeric x/y columns. JSON exports the complete experiment; CSV exports the selected table. Notebook results remain in memory until explicitly exported to `artifacts/tui/notebook/`. Import restores saved results and their recipes. Pin and Compare show descriptive differences for the same experiment.

Ready-to-run synthetic captures live in `examples/engine/canonical`, `examples/precision-b/path`, and `examples/two-decision/canonical`. Native execution is optional and needs a separately built upstream native wheel; NumPy works by default.

## Artwork and layout

`Ctrl+W` opens the original portrait; `Ctrl+B` toggles the surrounding navigation and inspector. Compact terminals reduce decoration. Artwork scrolls at its original character size. `GINSENG_MOTION=0` disables animation.

## Optional service connection

Connected application and API contracts are clients for an existing authenticated Ginseng server. No server is included or contacted at startup. Set `GINSENG_API_URL` and, when required, `GINSENG_API_TOKEN` in your shell before launching. Remote connections require HTTPS; the local default is `http://127.0.0.1:8000`.

Tokens stay in the environment, are omitted from recipes, and are not forwarded through redirects. GET reads data. PUT explicitly saves the complete edited snapshot with its expected revision; review the body before running it. Personal data is only exported locally when you choose an export action.

## Interpretation

Precision intervals measure numerical uncertainty under the specified simulator. Outer-bootstrap uncertainty and historical calibration are separate recipes. Cash-flow drawdowns are dollar losses, not investment returns. Portfolio and funding outputs compare hypothetical choices under declared assumptions.
