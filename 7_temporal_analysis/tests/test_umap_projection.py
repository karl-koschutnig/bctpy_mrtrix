"""
Test-driven development for 02_umap_projection.py

TDD Cycle: RED -> GREEN -> REFACTOR

These tests define the expected behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Load temporal configuration."""
    config_path = Path(__file__).parent.parent / "config" / "run_spec_temporal.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def sample_metrics_data():
    """Create sample metrics data for UMAP."""
    n_samples = 100
    return pd.DataFrame({
        'participant_id': [f'sub-{i:03d}' for i in range(n_samples)],
        'session': np.random.choice(['ses-1', 'ses-2', 'ses-3'], n_samples),
        'group': np.random.choice(['ctrl', '2w', '4w'], n_samples),
        'age': np.random.randint(20, 40, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
        'density': np.random.rand(n_samples),
        'global_efficiency': np.random.rand(n_samples),
        'charpath': np.random.rand(n_samples) * 10,
        'clustering_coef': np.random.rand(n_samples),
        'modularity': np.random.rand(n_samples),
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TESTS: run_umap function
# ============================================================================

class TestRunUMAP:
    """Tests for UMAP dimensionality reduction function."""

    def test_umap_returns_embedding(self, sample_metrics_data):
        """UMAP should return embedding with specified dimensions."""
        from scripts import umap_projection as umap_script
        
        # Select only numeric features
        feature_cols = ['density', 'global_efficiency', 'charpath', 'clustering_coef', 'modularity']
        
        embedding = umap_script.run_umap(sample_metrics_data[feature_cols], n_components=2)
        
        assert embedding.shape[0] == len(sample_metrics_data), "Embedding should have same number of rows"
        assert embedding.shape[1] == 2, "Embedding should have 2 dimensions"

    def test_umap_returns_3d_embedding(self, sample_metrics_data):
        """UMAP should return 3D embedding when requested."""
        from scripts import umap_projection as umap_script
        
        feature_cols = ['density', 'global_efficiency', 'charpath', 'clustering_coef', 'modularity']
        
        embedding = umap_script.run_umap(sample_metrics_data[feature_cols], n_components=3)
        
        assert embedding.shape[1] == 3, "Embedding should have 3 dimensions"

    def test_umap_reproducible_with_random_state(self, sample_metrics_data):
        """UMAP should be reproducible with same random state."""
        from scripts import umap_projection as umap_script
        
        feature_cols = ['density', 'global_efficiency', 'charpath', 'clustering_coef', 'modularity']
        
        embedding1 = umap_script.run_umap(sample_metrics_data[feature_cols], n_components=2, random_state=42)
        embedding2 = umap_script.run_umap(sample_metrics_data[feature_cols], n_components=2, random_state=42)
        
        assert np.allclose(embedding1, embedding2), "UMAP should be reproducible with same random state"

    def test_umap_handles_small_dataset(self):
        """UMAP should handle small datasets (e.g., 3 timepoints)."""
        from scripts import umap_projection as umap_script
        
        # Create minimal dataset with 3 samples
        data = pd.DataFrame({
            'metric1': [1.0, 2.0, 3.0],
            'metric2': [0.5, 0.6, 0.7],
        })
        
        embedding = umap_script.run_umap(data, n_components=2, n_neighbors=2)
        
        assert embedding.shape == (3, 2), "Should handle small dataset"


# ============================================================================
# TESTS: process_timepoint_averages function
# ============================================================================

class TestTimepointAverages:
    """Tests for timepoint averaging and trajectory analysis."""

    def test_timepoint_averages_returns_dataframe(self, sample_metrics_data):
        """Timepoint averages should return DataFrame."""
        from scripts import umap_projection as umap_script
        
        result = umap_script.calculate_timepoint_averages(
            sample_metrics_data,
            timepoint_col='session',
            metric_cols=['density', 'global_efficiency', 'charpath']
        )
        
        assert isinstance(result, pd.DataFrame), "Result should be DataFrame"
        assert 'session' in result.columns, "Result should have session column"

    def test_timepoint_averages_correct_mean(self, sample_metrics_data):
        """Timepoint averages should calculate correct means."""
        from scripts import umap_projection as umap_script
        
        # Filter to just density
        test_data = sample_metrics_data[['session', 'density']].copy()
        
        result = umap_script.calculate_timepoint_averages(
            test_data,
            timepoint_col='session',
            metric_cols=['density']
        )
        
        # Check that means are correct
        for _, row in result.iterrows():
            session = row['session']
            session_data = test_data[test_data['session'] == session]
            expected_mean = session_data['density'].mean()
            assert np.isclose(row['density'], expected_mean), f"Mean for {session} should be {expected_mean}"


# ============================================================================
# TESTS: turning point detection
# ============================================================================

class TestTurningPointDetection:
    """Tests for turning point detection."""

    def test_detect_turning_points_returns_indices(self):
        """Turning point detection should return indices."""
        from scripts import umap_projection as umap_script
        
        # Create embedding with clear turning point
        embedding = np.array([
            [0, 0],
            [1, 0],
            [5, 5],  # Turning point
            [6, 5],
        ])
        
        turning_indices = umap_script.detect_turning_points(embedding, gradient_threshold=0.5)
        
        assert isinstance(turning_indices, list), "Should return list of indices"
        assert 2 in turning_indices, "Should detect index 2 as turning point"

    def test_no_turning_points_smooth_trajectory(self):
        """Smooth trajectory should have no turning points."""
        from scripts import umap_projection as umap_script
        
        # Linear trajectory
        embedding = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        
        turning_indices = umap_script.detect_turning_points(embedding, gradient_threshold=0.5)
        
        assert len(turning_indices) == 0, "Smooth trajectory should have no turning points"


# ============================================================================
# TESTS: Main function and CLI
# ============================================================================

class TestMainFunction:
    """Tests for main function and CLI."""

    def test_main_runs_without_error(self, sample_metrics_data, temp_output_dir):
        """Main function should run without errors."""
        from scripts import umap_projection as umap_script
        
        # Save test data
        input_file = temp_output_dir / "input_metrics.csv"
        sample_metrics_data.to_csv(input_file, index=False)
        
        output_dir = temp_output_dir / "umap_output"
        
        # Mock sys.argv
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                '02_umap_projection.py',
                '--input-file', str(input_file),
                '--output-dir', str(output_dir),
                '--timepoint-col', 'session',
                '--n-components', '2'
            ]
            
            # Should not raise
            umap_script.main()
            
        finally:
            sys.argv = original_argv

    def test_main_creates_output_files(self, sample_metrics_data, temp_output_dir):
        """Main function should create output files."""
        from scripts import umap_projection as umap_script
        
        input_file = temp_output_dir / "input_metrics.csv"
        sample_metrics_data.to_csv(input_file, index=False)
        
        output_dir = temp_output_dir / "umap_output"
        
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                '02_umap_projection.py',
                '--input-file', str(input_file),
                '--output-dir', str(output_dir),
                '--timepoint-col', 'session',
                '--n-components', '2'
            ]
            
            umap_script.main()
            
            # Check output files exist
            assert (output_dir / "umap_coordinates.csv").exists(), "Should create UMAP coordinates file"
            
        finally:
            sys.argv = original_argv


# ============================================================================
# TESTS: Visualization
# ============================================================================

class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_umap_trajectory_runs(self, sample_metrics_data, temp_output_dir):
        """UMAP trajectory plot should run without errors."""
        from scripts import umap_projection as umap_script
        
        # Create embedding
        feature_cols = ['density', 'global_efficiency', 'charpath', 'clustering_coef', 'modularity']
        embedding = umap_script.run_umap(sample_metrics_data[feature_cols], n_components=2)
        
        sample_metrics_data['umap_1'] = embedding[:, 0]
        sample_metrics_data['umap_2'] = embedding[:, 1]
        
        plot_path = temp_output_dir / "umap_plot.png"
        
        # Should not raise
        umap_script.plot_umap_trajectory(
            sample_metrics_data,
            umap_cols=['umap_1', 'umap_2'],
            color_col='session',
            output_path=str(plot_path)
        )
        
        assert plot_path.exists(), "Should create plot file"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
