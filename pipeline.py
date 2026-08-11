"""
pipeline.py
-----------
End-to-end orchestration: runs Modules 1-4 in sequence.
This is the single entry point you'd call from an API/scheduler in production.
"""

import subprocess
import sys

STEPS = [
    ("Generating simulated multi-source data", "generate_data.py"),
    ("Building fused feature tables", "build_features.py"),
    ("Running stress detection (rule-based + ML)", "stress_detection.py"),
    ("Training image branch CNN", "image_branch.py"),
    ("Fusing image + sensor data (multimodal)", "fusion_model.py"),
    ("Running yield prediction (RF vs XGBoost)", "yield_prediction.py"),
    ("Generating farmer advisories", "advisory_system.py"),
]

if __name__ == "__main__":
    for label, script in STEPS:
        print(f"\n{'='*60}\nSTEP: {label}\n{'='*60}")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Pipeline failed at: {script}")
            sys.exit(1)
    print("\nPipeline completed successfully. Check ../outputs/ for results.")
