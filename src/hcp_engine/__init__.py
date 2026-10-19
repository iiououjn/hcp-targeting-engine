"""HCP segmentation & targeting engine.

Behavioral clustering of healthcare professionals (HCPs) plus a SHAP-explained
propensity-to-prescribe model, combined into an actionable targeting table.
"""

from hcp_engine.config import Config
from hcp_engine.data import HCPData, generate_synthetic, load_csv_dir, save_csv_dir
from hcp_engine.pipeline import PipelineResult, run_pipeline

__version__ = "0.1.0"

__all__ = [
    "Config",
    "HCPData",
    "PipelineResult",
    "generate_synthetic",
    "load_csv_dir",
    "run_pipeline",
    "save_csv_dir",
]
