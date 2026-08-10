import streamlit as st

def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
        .pipeline {
            display:flex; align-items:center; gap:10px; flex-wrap:wrap;
            padding:18px; border:1px solid rgba(128,128,128,.25);
            border-radius:18px; margin:12px 0 24px 0;
        }
        .node {
            padding:14px 16px; border-radius:14px;
            border:1px solid rgba(128,128,128,.32);
            min-width:145px; text-align:center; font-weight:650;
        }
        .node.hot {border-width:2px;}
        .node.good {border-width:2px;}
        .arrow {font-size:25px; opacity:.65;}
        .badge {
            display:inline-block; padding:5px 10px; border-radius:999px;
            font-size:.78rem; font-weight:800; letter-spacing:.04em;
            border:1px solid rgba(128,128,128,.35); margin-bottom:12px;
        }
        .badge.warn {border-width:2px;}
        .badge.good {border-width:2px;}
        .badge.info {border-width:1px;}
        small {font-weight:400; opacity:.75;}
        </style>
        """,
        unsafe_allow_html=True,
    )
