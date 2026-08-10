import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
from utils.demo_pipeline import run_demo_pipeline, make_overlay
from utils.styles import inject_css

st.set_page_config(
    page_title="MAPVNet-Q Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

if "runs" not in st.session_state:
    st.session_state.runs = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.sidebar.markdown("## ☀️ MAPVNet-Q")
st.sidebar.caption("Quality-Aware Multi-Agent PV Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Live Inference",
        "Agent Decision",
        "Quality Analysis",
        "Audit & Verify",
        "BDAPV Acquire & Admit",
        "Research Rounds",
        "Settings",
    ],
)

st.sidebar.markdown("---")
mode = st.sidebar.radio("System mode", ["Demo Mode", "Research Mode"], index=0)
st.sidebar.caption(
    "Demo Mode uses simulated outputs. Research Mode exposes additional diagnostic values."
)

def badge(text, kind="info"):
    st.markdown(f'<span class="badge {kind}">{text}</span>', unsafe_allow_html=True)

def title_block(title, subtitle):
    st.markdown(f"# {title}")
    st.caption(subtitle)

def no_run_message():
    st.info("Run an image first from **Live Inference** to populate this page.")

if page == "Overview":
    title_block(
        "MAPVNet-Q Research Dashboard",
        "Interactive demonstrator for uncertainty-driven multi-agent photovoltaic segmentation.",
    )

    st.warning(
        "VERSION 1 STATUS — This dashboard is a research UI prototype. "
        "Inference, uncertainty, decision, fusion, audit, and acquisition values are currently DEMO/SIMULATED "
        "until your real MAPVNet-Q modules are connected."
    )

    a, b, c, d = st.columns(4)
    a.metric("System", "READY")
    b.metric("Agents", "3")
    c.metric("Decision actions", "4")
    d.metric("Research rounds", "5")

    st.markdown("### End-to-end pipeline")
    st.markdown(
        """
        <div class="pipeline">
          <div class="node">Input image<br><small>JPG / PNG / TIFF</small></div>
          <div class="arrow">→</div>
          <div class="node">Pre-processing<br><small>normalize / resize</small></div>
          <div class="arrow">→</div>
          <div class="node">Multi-resolution agents<br><small>0.8 m · 0.3 m · 0.1 m</small></div>
          <div class="arrow">→</div>
          <div class="node">Quality engine<br><small>entropy · TTA · boundary · Pred-IoU</small></div>
          <div class="arrow">→</div>
          <div class="node hot">Decision agent<br><small>accept · retry · re-route · escalate</small></div>
          <div class="arrow">→</div>
          <div class="node">Adaptive fusion</div>
          <div class="arrow">→</div>
          <div class="node">Instance extraction</div>
          <div class="arrow">→</div>
          <div class="node good">Final PV result</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Continual research loop")
    st.markdown(
        """
        <div class="pipeline">
          <div class="node">AUDIT</div><div class="arrow">→</div>
          <div class="node">VERIFY</div><div class="arrow">→</div>
          <div class="node">ACQUIRE</div><div class="arrow">→</div>
          <div class="node">ADMIT / REJECT</div><div class="arrow">→</div>
          <div class="node hot">NEXT ROUND</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What becomes real later")
    st.write(
        "Replace the demo functions in `utils/demo_pipeline.py` with your actual PyTorch segmentation agents, "
        "uncertainty estimator, decision policy, adaptive fusion, and constrained instance extraction. "
        "The dashboard structure can remain the same."
    )

elif page == "Live Inference":
    title_block(
        "Live Inference",
        "Upload a PV image and execute the full MAPVNet-Q demonstrator pipeline.",
    )
    badge("DEMO OUTPUTS", "warn")

    uploaded = st.file_uploader(
        "Upload PV image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="For Version 1, use a normal RGB image. Very large TIFF files should be downsampled first.",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        arr = np.array(image)

        left, right = st.columns([1.15, 0.85])
        with left:
            st.image(image, caption="Input image", use_container_width=True)
        with right:
            st.markdown("### Image metadata")
            st.write(f"**File:** {uploaded.name}")
            st.write(f"**Resolution:** {image.width} × {image.height}")
            st.write(f"**Channels:** RGB")
            gsd = st.selectbox("Declared GSD / sensing scale", ["Unknown", "0.8 m", "0.3 m", "0.1 m"])
            source = st.selectbox("Source", ["Unknown", "Satellite", "Aerial", "UAV / Drone"])

        if st.button("▶ RUN MAPVNet-Q", type="primary", use_container_width=True):
            stages = [
                "Pre-processing",
                "0.8 m agent",
                "0.3 m agent",
                "0.1 m agent",
                "Quality estimation",
                "Decision policy",
                "Adaptive fusion",
                "Instance extraction",
                "Final result",
            ]
            progress = st.progress(0)
            stage_box = st.empty()

            for i, s in enumerate(stages, start=1):
                stage_box.info(f"Running: {s}")
                progress.progress(i / len(stages))

            result = run_demo_pipeline(arr, declared_gsd=gsd, source=source)
            result["filename"] = uploaded.name
            result["input_image"] = arr
            st.session_state.last_result = result
            st.session_state.runs.append({
                "file": uploaded.name,
                "quality": result["quality"]["quality_score"],
                "decision": result["decision"]["action"],
                "pred_iou": result["quality"]["pred_iou"],
                "modules": result["instances"]["valid_modules"],
            })
            stage_box.success("Pipeline complete")
            st.rerun()

    result = st.session_state.last_result
    if result is not None:
        st.markdown("---")
        st.markdown("## Latest result")

        q = result["quality"]
        d = result["decision"]
        inst = result["instances"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Quality", f'{q["quality_score"]:.3f}')
        c2.metric("Predicted IoU", f'{q["pred_iou"]:.3f}')
        c3.metric("Decision", d["action"])
        c4.metric("Valid modules", inst["valid_modules"])
        c5.metric("Estimated PV area", f'{inst["pv_area_m2"]:.1f} m²')

        st.markdown("### Agent predictions")
        p1, p2, p3 = st.columns(3)
        p1.image(result["agents"]["0.8m"], caption="0.8 m agent — demo mask", use_container_width=True)
        p2.image(result["agents"]["0.3m"], caption="0.3 m agent — demo mask", use_container_width=True)
        p3.image(result["agents"]["0.1m"], caption="0.1 m agent — demo mask", use_container_width=True)

        st.markdown("### Final result")
        a, b, c = st.columns(3)
        a.image(result["input_image"], caption="Original", use_container_width=True)
        b.image(result["final_mask"], caption="Fused mask — demo", use_container_width=True)
        c.image(
            make_overlay(result["input_image"], result["final_mask"]),
            caption="Overlay — demo",
            use_container_width=True,
        )

        with st.expander("Decision history", expanded=True):
            st.dataframe(pd.DataFrame(result["history"]), use_container_width=True, hide_index=True)

elif page == "Agent Decision":
    title_block("Agent Decision", "Inspect why the quality-aware policy selected its action.")
    result = st.session_state.last_result
    if result is None:
        no_run_message()
    else:
        q, d = result["quality"], result["decision"]
        st.warning("Decision values are simulated in Version 1.")

        left, right = st.columns([0.9, 1.1])
        with left:
            st.markdown("### Decision card")
            kind = "good" if d["action"] == "ACCEPT" else "warn"
            badge(d["action"], kind)
            st.metric("Overall quality", f'{q["quality_score"]:.3f}')
            st.metric("Retry count", f'{d["retry_count"]} / 3')
            st.metric("Compute budget used", f'{d["compute_budget_pct"]}%')
            st.write(f"**Reason:** {d['reason']}")
            st.write(f"**Route:** {d['route']}")

        with right:
            st.markdown("### Four-action policy")
            policy_df = pd.DataFrame([
                {"Action": "ACCEPT", "Meaning": "Quality threshold satisfied", "Active": d["action"] == "ACCEPT"},
                {"Action": "RETRY", "Meaning": "Repeat inference / enhanced preprocessing", "Active": d["action"] == "RETRY"},
                {"Action": "RE-ROUTE", "Meaning": "Send tile to a specialist agent", "Active": d["action"] == "RE-ROUTE"},
                {"Action": "ESCALATE", "Meaning": "Flag unresolved hard case", "Active": d["action"] == "ESCALATE"},
            ])
            st.dataframe(policy_df, use_container_width=True, hide_index=True)

        st.markdown("### Decision trace")
        st.dataframe(pd.DataFrame(result["history"]), use_container_width=True, hide_index=True)

elif page == "Quality Analysis":
    title_block(
        "Quality Analysis",
        "Uncertainty and quality signals used by the MAPVNet-Q decision policy.",
    )
    result = st.session_state.last_result
    if result is None:
        no_run_message()
    else:
        q = result["quality"]
        st.warning("Quality values are simulated in Version 1.")

        metrics = pd.DataFrame({
            "Metric": ["Entropy", "TTA variance", "Boundary disagreement", "Predicted IoU"],
            "Value": [q["entropy"], q["tta_variance"], q["boundary_disagreement"], q["pred_iou"]],
        })
        fig = px.bar(metrics, x="Metric", y="Value", range_y=[0, 1], text_auto=".3f")
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entropy ↓", f'{q["entropy"]:.3f}')
        c2.metric("TTA variance ↓", f'{q["tta_variance"]:.3f}')
        c3.metric("Boundary disagreement ↓", f'{q["boundary_disagreement"]:.3f}')
        c4.metric("Pred-IoU ↑", f'{q["pred_iou"]:.3f}')

        st.markdown("### Adaptive fusion weights")
        weights = pd.DataFrame({
            "Agent": ["0.8 m", "0.3 m", "0.1 m"],
            "Weight": result["fusion_weights"],
        })
        st.dataframe(weights, use_container_width=True, hide_index=True)
        fig2 = px.bar(weights, x="Agent", y="Weight", range_y=[0, 1], text_auto=".2f")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Audit & Verify":
    title_block(
        "Audit & Verify",
        "Identify suspicious predictions and verify whether they represent real model failures.",
    )
    result = st.session_state.last_result
    if result is None:
        no_run_message()
    else:
        st.warning("Audit rules and values are simulated in Version 1.")
        q = result["quality"]

        audit_flags = {
            "High entropy": q["entropy"] > 0.45,
            "High TTA disagreement": q["tta_variance"] > 0.30,
            "High boundary disagreement": q["boundary_disagreement"] > 0.35,
            "Low Predicted IoU": q["pred_iou"] < 0.70,
        }
        flags = sum(audit_flags.values())
        confirmed = flags >= 2

        a, b = st.columns([1, 1])
        with a:
            st.markdown("### AUDIT")
            for name, active in audit_flags.items():
                st.write(("⚠️ " if active else "✅ ") + name)
        with b:
            st.markdown("### VERIFY")
            if confirmed:
                st.error("FAILURE CONFIRMED — candidate for acquisition.")
            else:
                st.success("No verified hard failure under the current demo rules.")
            st.metric("Triggered failure signals", f"{flags} / 4")

        if confirmed:
            st.markdown("### Failure signature")
            st.json({
                "tile": result.get("filename", "current tile"),
                "dominant_issue": "uncertainty / boundary instability",
                "recommended_next_step": "search external dataset for relevant candidates",
                "target_dataset": "BDAPV",
            })

elif page == "BDAPV Acquire & Admit":
    title_block(
        "BDAPV Acquire & Admit",
        "Prototype interface for searching an external candidate pool and admitting useful samples.",
    )
    st.warning(
        "No BDAPV dataset is connected in Version 1. The candidate records below are synthetic UI examples only."
    )

    demo_candidates = pd.DataFrame([
        {"Candidate": "DEMO-BD-001", "Similarity": 0.94, "Novelty": 0.82, "Image quality": 0.91, "Failure relevance": 0.93},
        {"Candidate": "DEMO-BD-002", "Similarity": 0.88, "Novelty": 0.91, "Image quality": 0.86, "Failure relevance": 0.89},
        {"Candidate": "DEMO-BD-003", "Similarity": 0.79, "Novelty": 0.76, "Image quality": 0.95, "Failure relevance": 0.72},
        {"Candidate": "DEMO-BD-004", "Similarity": 0.67, "Novelty": 0.94, "Image quality": 0.84, "Failure relevance": 0.61},
    ])
    st.dataframe(demo_candidates, use_container_width=True, hide_index=True)

    selected = st.selectbox("Candidate to review", demo_candidates["Candidate"])
    row = demo_candidates[demo_candidates["Candidate"] == selected].iloc[0]
    score = float(np.mean([row["Similarity"], row["Novelty"], row["Image quality"], row["Failure relevance"]]))
    st.metric("Composite admission score", f"{score:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✓ ADMIT", type="primary", use_container_width=True):
            st.success(f"{selected} marked ADMIT in this demo session.")
    with c2:
        if st.button("✕ REJECT", use_container_width=True):
            st.info(f"{selected} marked REJECT in this demo session.")

    st.caption(
        "Real integration: replace this table with embeddings / similarity search over your actual BDAPV candidate pool, "
        "then record admitted sample IDs and provenance."
    )

elif page == "Research Rounds":
    title_block(
        "Five Research Rounds",
        "Track iterative audit → verify → acquire → admit → retrain/evaluate cycles.",
    )
    st.warning("The round values below are demonstration placeholders, not experimental results.")

    rounds = pd.DataFrame({
        "Round": [1, 2, 3, 4, 5],
        "Verified failures": [42, 35, 27, 19, 13],
        "Admitted samples": [80, 67, 54, 41, 29],
        "Demo mIoU": [0.78, 0.81, 0.835, 0.852, 0.865],
        "Demo retry success": [0.52, 0.59, 0.65, 0.70, 0.74],
    })
    st.dataframe(rounds, use_container_width=True, hide_index=True)

    fig = px.line(rounds, x="Round", y=["Demo mIoU", "Demo retry success"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Intended experimental loop")
    st.write(
        "**Round n:** run model → audit failures → verify genuine failures → acquire relevant candidates → "
        "admit useful samples → retrain/fine-tune → evaluate on the fixed held-out protocol → proceed to next round."
    )

elif page == "Settings":
    title_block("Settings", "Prototype controls for thresholds and future model integration.")

    st.markdown("### Demo policy thresholds")
    st.slider("Accept quality threshold", 0.0, 1.0, 0.78, 0.01, disabled=True)
    st.slider("Escalation quality threshold", 0.0, 1.0, 0.42, 0.01, disabled=True)
    st.number_input("Maximum retries", min_value=0, max_value=10, value=3, disabled=True)

    st.markdown("### Real model connection checklist")
    st.checkbox("0.8 m segmentation agent connected", value=False, disabled=True)
    st.checkbox("0.3 m segmentation agent connected", value=False, disabled=True)
    st.checkbox("0.1 m segmentation agent connected", value=False, disabled=True)
    st.checkbox("Quality estimator connected", value=False, disabled=True)
    st.checkbox("Decision policy connected", value=False, disabled=True)
    st.checkbox("Adaptive fusion connected", value=False, disabled=True)
    st.checkbox("Instance extraction connected", value=False, disabled=True)
    st.checkbox("BDAPV retrieval connected", value=False, disabled=True)

if mode == "Research Mode":
    with st.sidebar.expander("Research diagnostics"):
        st.write(f"Completed demo runs: {len(st.session_state.runs)}")
        if st.session_state.last_result is not None:
            st.json({
                "decision": st.session_state.last_result["decision"]["action"],
                "quality": st.session_state.last_result["quality"]["quality_score"],
                "weights": st.session_state.last_result["fusion_weights"],
            })
