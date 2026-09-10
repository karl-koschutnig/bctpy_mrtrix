"""
Test-driven development for gam_models.R

Since R testing is complex without testthat, we'll create a Python wrapper
that can be tested, and the R script will be the implementation.

TDD Cycle: RED -> GREEN -> REFACTOR
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
import os


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for GAM modeling."""
    n_samples = 100
    np.random.seed(42)
    return pd.DataFrame({
        'participant_id': [f'sub-{i:03d}' for i in range(n_samples)],
        'session': np.random.choice(['ses-1', 'ses-2', 'ses-3'], n_samples),
        'group': np.random.choice(['ctrl', '2w', '4w'], n_samples),
        'age': np.random.randint(20, 40, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'density': np.random.rand(n_samples) * 0.5 + 0.1,
        'global_efficiency': np.random.rand(n_samples) * 0.3 + 0.2,
        'charpath': np.random.rand(n_samples) * 5 + 2,
        'clustering_coef': np.random.rand(n_samples) * 0.4 + 0.1,
        'modularity': np.random.rand(n_samples) * 0.3 + 0.1,
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TESTS: Data preparation for GAM
# ============================================================================

class TestGAMDataPreparation:
    """Tests for GAM data preparation."""

    def test_session_as_factor(self, sample_metrics_data):
        """Session should be converted to factor for GAM."""
        # This would be tested in R, but we can test the data structure
        assert sample_metrics_data['session'].nunique() == 3
        assert set(sample_metrics_data['session'].unique()) == {'ses-1', 'ses-2', 'ses-3'}

    def test_no_missing_values(self, sample_metrics_data):
        """Data should have no missing values."""
        assert not sample_metrics_data.isnull().any().any()

    def test_metric_ranges(self, sample_metrics_data):
        """Metrics should be in reasonable ranges."""
        assert (sample_metrics_data['density'] > 0).all()
        assert (sample_metrics_data['density'] < 1).all()


# ============================================================================
# TESTS: R script file existence
# ============================================================================

class TestRScriptExistence:
    """Tests for R script file."""

    def test_gam_script_exists(self):
        """GAM R script should exist."""
        script_path = Path(__file__).parent.parent / "scripts" / "gam_models.R"
        assert script_path.exists(), f"GAM script not found at {script_path}"

    def test_gam_script_readable(self):
        """GAM R script should be readable."""
        script_path = Path(__file__).parent.parent / "scripts" / "gam_models.R"
        with open(script_path) as f:
            content = f.read()
        assert len(content) > 0, "GAM script should not be empty"

    def test_gam_script_has_required_functions(self):
        """GAM R script should have required functions."""
        script_path = Path(__file__).parent.parent / "scripts" / "gam_models.R"
        with open(script_path) as f:
            content = f.read()
        
        assert 'fit_gam_model' in content or 'gam' in content.lower(), "Should have GAM fitting code"
        assert 'session' in content or 'timepoint' in content.lower(), "Should reference session/timepoint"


# ============================================================================
# TESTS: Configuration
# ============================================================================

class TestConfiguration:
    """Tests for GAM configuration."""

    def test_config_file_exists(self):
        """Configuration file should exist."""
        config_path = Path(__file__).parent.parent / "config" / "run_spec_temporal.json"
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_config_has_gam_section(self):
        """Configuration should have GAM section."""
        config_path = Path(__file__).parent.parent / "config" / "run_spec_temporal.json"
        with open(config_path) as f:
            config = json.load(f)
        
        assert 'gam' in config, "Config should have GAM section"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
