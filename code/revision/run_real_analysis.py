"""Runs the real-data analysis in order (run from code/revision/; data location from LINKAGE_DATA_DIR, see ../common.py).
Each step writes only under outputs/revision/mim_final/.
  1 extract_neonatal_sim.py      rebuilds the neonatal extracts from the general SIM files and compares them with preserved extracts
  2 audit_mim_inputs.py          manifest of inputs (hashes, versions)
  3 linkage_final.py             annual runs with identities, consolidation, four configurations, flow counts
  4 task_h_real_sensitivity.py   yield, transitions, strata
  5 task_i_model_concordance.py  logistic model and agreement on fields not used in linkage
  6 audit_exact_ambiguity.py     ambiguity of the exact passes and multiple pregnancy by stage
"""
import subprocess, sys
for s in ["extract_neonatal_sim.py", "audit_mim_inputs.py", "linkage_final.py", "task_h_real_sensitivity.py",
          "task_i_model_concordance.py", "audit_exact_ambiguity.py"]:
    print("==", s, flush=True)
    subprocess.run([sys.executable, s], check=True)
