import streamlit as st
import subprocess
import sys
import os

st.set_page_config(page_title="Recall-Aware IDS SOC Dashboard", layout="wide")

st.title("Recall-Aware IDS SOC Dashboard")
st.markdown("""
This dashboard simulates a Security Operations Center (SOC) view of the Recall-Aware IDS.
It demonstrates the behavioral difference between the existing defense (Base) and the defense controlled by Recall-Aware.
""")

st.sidebar.header("Demonstration Settings")
attack = st.sidebar.selectbox("Attack Scenario", ["silent_probing", "surrogate_transfer", "decision_boundary", "none"])
defense = st.sidebar.selectbox("Defense Mechanism", ["afp", "rs", "fs", "none"])
controller = st.sidebar.selectbox("Controller State", ["base", "recall-aware"])

compare = st.sidebar.checkbox("Run Comparison Mode (No Defense vs Base vs Recall-Aware)")

if st.sidebar.button("Run Simulation"):
    cmd = [sys.executable, "run.py", "--attack", attack, "--defense", defense, "--controller", controller]
    if compare:
        cmd.append("--compare")
        
    with st.spinner("Running simulation..."):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.success("Simulation Complete")
            st.text_area("Console Output", result.stdout, height=400)
        except subprocess.CalledProcessError as e:
            st.error("Simulation Failed")
            st.text_area("Error Output", e.stderr, height=400)
