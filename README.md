# MAPVNet-Q Dashboard

Interactive Streamlit research dashboard for the proposed MAPVNet-Q workflow.

## Version 1 status

This is a UI / workflow prototype. All segmentation masks, uncertainty metrics,
decision outputs, fusion weights, audit flags, BDAPV candidates, and research-round
numbers are explicitly DEMO/SIMULATED unless you replace the demo pipeline with your
real research implementation.

## Pipeline represented

Input -> preprocessing -> 0.8m / 0.3m / 0.1m agents -> quality estimation
(entropy, TTA variance, boundary disagreement, predicted IoU) -> four-action
decision policy (ACCEPT / RETRY / RE-ROUTE / ESCALATE) -> adaptive fusion ->
instance extraction -> final PV result.

A second research loop is represented as:
AUDIT -> VERIFY -> ACQUIRE from BDAPV -> ADMIT / REJECT -> next research round.

## Files

- `app.py`: dashboard and navigation
- `utils/demo_pipeline.py`: demo pipeline; replace with real inference later
- `utils/styles.py`: UI styling
- `requirements.txt`: Python dependencies

## Run locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install:
```bash
pip install -r requirements.txt
```

Run:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload all files and folders to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create an app.
4. Select the repository and `main` branch.
5. Set the entrypoint file to `app.py`.
6. Deploy.

## Connecting the real MAPVNet-Q model

Replace `run_demo_pipeline()` in `utils/demo_pipeline.py` with wrappers around:
1. preprocessing
2. 0.8m agent
3. 0.3m agent
4. 0.1m agent
5. entropy / TTA / boundary / Pred-IoU quality estimator
6. ACCEPT / RETRY / RE-ROUTE / ESCALATE policy
7. adaptive fusion
8. constrained instance extraction

Keep the return dictionary keys the same and the UI can remain almost unchanged.
