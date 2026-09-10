"""
Test-driven development for turning_points.py

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
def sample_umap_data():
    """Create sample UMAP data for testing."""
    n_samples = 100
    np.random.seed(42)
    
    # Create a trajectory that has clear turning points
    # First 30 samples: moving in one direction
    # Next 30 samples: turning
    # Last 40 samples: moving in new direction
    
    data = pd.DataFrame({
        'participant_id': [f'sub-{i:03d}' for i in range(n_samples)],
        'session': [f'ses-{(i % 3) + 1}' for i in range(n_samples)],
        'group': np.random.choice(['ctrl', '2w', '4w'], n_samples),
        'age': np.random.randint(20, 40, n_samples),
        'sex': np.random.choice([0, 1], n_samples),
    })
    
    # Create UMAP coordinates with turning points
    # Linear trajectory for ses-1, turn at ses-2, different direction for ses-3
    data['umap_1'] = np.zeros(n_samples)
    data['umap_2'] = np.zeros(n_samples)
    
    for i in range(n_samples):
        session = data['session'].iloc[i]
        if session == 'ses-1':
            data.at[i, 'umap_1'] = i / 10.0
            data.at[i, 'umap_2'] = i / 10.0
        elif session == 'ses-2':
            data.at[i, 'umap_1'] = 3 + (i % 30) / 10.0
            data.at[i, 'umap_2'] = 3 - (i % 30) / 10.0  # Turn here
        else:  # ses-3
            data.at[i, 'umap_1'] = 6 + (i % 40) / 10.0
            data.at[i, 'umap_2'] = 0 - (i % 40) / 10.0
    
    return data


@pytest.fixture
def linear_trajectory_data():
    """Create data with a smooth linear trajectory (no turning points)."""
    n_samples = 90
    data = pd.DataFrame({
        'participant_id': [f'sub-{i:03d}' for i in range(n_samples)],
        'session': [f'ses-{(i // 30) + 1}' for i in range(n_samples)],
        'umap_1': np.linspace(0, 10, n_samples),
        'umap_2': np.linspace(0, 10, n_samples),
    })
    return data


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# TESTS: detect_turning_points function
# ============================================================================

class TestDetectTurningPoints:
    """Tests for turning point detection function."""

    def test_detects_turning_point_in_simple_trajectory(self):
        """Should detect turning point in a simple L-shaped trajectory."""
        from scripts.turning_points import detect_turning_points
        
        # Create L-shaped trajectory: (0,0) -> (1,0) -> (1,1)
        embedding = np.array([
            [0, 0],
            [0.5, 0],
            [1, 0],
            [1, 0.5],
            [1, 1],
        ])
        
        turning_indices = detect_turning_points(embedding, gradient_threshold=0.5)
        
        # Should detect index 2 (the corner)
        assert 2 in turning_indices, f"Should detect turning point at index 2, got {turning_indices}"

    def test_no_turning_points_in_straight_line(self):
        """Straight line should have no turning points."""
        from scripts.turning_points import detect_turning_points
        
        # Linear trajectory
        embedding = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ])
        
        turning_indices = detect_turning_points(embedding, gradient_threshold=0.5)
        
        assert len(turning_indices) == 0, f"Straight line should have no turning points, got {turning_indices}"

    def test_too_few_points_returns_empty(self):
        """Less than 3 points should return empty list."""
        from scripts.turning_points import detect_turning_points
        
        # Only 2 points
        embedding = np.array([[0, 0], [1, 1]])
        
        turning_indices = detect_turning_points(embedding)
        
        assert turning_indices == [], "Should return empty list for < 3 points"

    def test_1d_embedding_returns_empty(self):
        """1D embedding should return empty list."""
        from scripts.turning_points import detect_turning_points
        
        # 1D embedding (only 1 column)
        embedding = np.array([[0], [1], [2], [3]])
        
        turning_indices = detect_turning_points(embedding)
        
        assert turning_indices == [], "Should return empty list for 1D embedding"

    def test_threshold_affects_detection(self):
        """Higher threshold should detect fewer turning points."""
        from scripts.turning_points import detect_turning_points
        
        # Create trajectory with a gentle turn
        embedding = np.array([
            [0, 0],
            [1, 0.1],
            [2, 0.5],  # Gentle turn
            [3, 1],
        ])
        
        # With low threshold, should detect the turn
        low_threshold_indices = detect_turning_points(embedding, gradient_threshold=0.1)
        
        # With high threshold, should not detect
        high_threshold_indices = detect_turning_points(embedding, gradient_threshold=1.5)
        
        # Low threshold should detect more (or equal) turning points
        assert len(low_threshold_indices) >= len(high_threshold_indices)


# ============================================================================
# TESTS: identify_epochs function
# ============================================================================

class TestIdentifyEpochs:
    """Tests for epoch identification function."""

    def test_identifies_epochs_from_turning_points(self):
        """Should correctly divide timeline into epochs based on turning points."""
        from scripts.turning_points import identify_epochs
        
        n_timepoints = 10
        turning_indices = [2, 5, 7]
        
        epochs = identify_epochs(turning_indices, n_timepoints)
        
        # Should create 4 epochs: [0,1,2], [3,4,5], [6,7], [8,9]
        assert len(epochs) == 4, f"Should create 4 epochs, got {len(epochs)}"
        assert epochs[0] == [0, 1, 2], f"First epoch should be [0,1,2], got {epochs[0]}"
        assert epochs[1] == [3, 4, 5], f"Second epoch should be [3,4,5], got {epochs[1]}"
        assert epochs[2] == [6, 7], f"Third epoch should be [6,7], got {epochs[2]}"
        assert epochs[3] == [8, 9], f"Fourth epoch should be [8,9], got {epochs[3]}"

    def test_no_turning_points_single_epoch(self):
        """No turning points should result in a single epoch."""
        from scripts.turning_points import identify_epochs
        
        n_timepoints = 10
        turning_indices = []
        
        epochs = identify_epochs(turning_indices, n_timepoints)
        
        assert len(epochs) == 1, "Should create 1 epoch with no turning points"
        assert epochs[0] == list(range(10)), "Single epoch should contain all timepoints"

    def test_turning_point_at_boundary(self):
        """Turning point at boundary should be handled correctly."""
        from scripts.turning_points import identify_epochs
        
        n_timepoints = 10
        turning_indices = [0, 9]  # At first and last
        
        epochs = identify_epochs(turning_indices, n_timepoints)
        
        # Should create 3 epochs
        assert len(epochs) == 3


# ============================================================================
# TESTS: Main processing function
# ============================================================================

class TestMainProcessing:
    """Tests for main processing function."""

    def test_process_turning_points_returns_dict(self, sample_umap_data, temp_output_dir):
        """Processing should return dictionary with results."""
        from scripts.turning_points import process_turning_points
        
        input_file = temp_output_dir / "input.csv"
        sample_umap_data.to_csv(input_file, index=False)
        
        results = process_turning_points(
            input_file=str(input_file),
            output_dir=str(temp_output_dir / "output"),
            timepoint_col='session'
        )
        
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'turning_points' in results, "Results should have turning_points"
        assert 'turning_indices' in results, "Results should have turning_indices"
        assert 'epochs' in results, "Results should have epochs"

    def test_process_creates_output_files(self, sample_umap_data, temp_output_dir):
        """Processing should create output files."""
        from scripts.turning_points import process_turning_points
        
        input_file = temp_output_dir / "input.csv"
        sample_umap_data.to_csv(input_file, index=False)
        
        output_dir = temp_output_dir / "output"
        
        process_turning_points(
            input_file=str(input_file),
            output_dir=str(output_dir),
            timepoint_col='session'
        )
        
        assert (output_dir / "turning_points.json").exists(), "Should create turning_points.json"
        assert (output_dir / "data_with_epochs.csv").exists(), "Should create data_with_epochs.csv"

    def test_process_handles_missing_columns(self):
        """Should handle missing UMAP columns gracefully."""
        from scripts.turning_points import process_turning_points
        
        # Create data without UMAP columns
        data = pd.DataFrame({
            'participant_id': ['sub-001', 'sub-002'],
            'session': ['ses-1', 'ses-2'],
        })
        
        input_file = Path(tempfile.mktemp(suffix='.csv'))
        data.to_csv(input_file, index=False)
        
        output_dir = Path(tempfile.mkdtemp())
        
        # Should raise error
        with pytest.raises((ValueError, KeyError)):
            process_turning_points(
                input_file=str(input_file),
                output_dir=str(output_dir),
                timepoint_col='session'
            )


# ============================================================================
# TESTS: Visualization functions
# ============================================================================

class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_functions_dont_crash(self, sample_umap_data, temp_output_dir):
        """Plot functions should not crash with valid data."""
        from scripts.turning_points import (
            plot_trajectory_with_turning_points,
            plot_epochs
        )
        
        # These should not raise even without matplotlib
        try:
            plot_trajectory_with_turning_points(
                sample_umap_data,
                umap_cols=['umap_1', 'umap_2'],
                timepoint_col='session',
                turning_points=['ses-2'],
                output_path=str(temp_output_dir / "test_plot.png")
            )
        except Exception as e:
            # Should only warn, not raise
            assert "Matplotlib" in str(e) or "skip" in str(e).lower()
        
        try:
            plot_epochs(
                sample_umap_data,
                epochs=[['ses-1'], ['ses-2'], ['ses-3']],
                umap_cols=['umap_1', 'umap_2'],
                timepoint_col='session',
                output_path=str(temp_output_dir / "test_epochs.png")
            )
        except Exception as e:
            assert "Matplotlib" in str(e) or "skip" in str(e).lower()


# ============================================================================
# TESTS: Epoch analysis functions
# ============================================================================

class TestEpochAnalysis:
    """Tests for epoch analysis functions."""

    def test_pca_per_epoch_returns_dict(self, sample_umap_data):
        """PCA per epoch should return dictionary."""
        from scripts.turning_points import pca_per_epoch
        
        epochs = [['ses-1'], ['ses-2'], ['ses-3']]
        
        try:
            results = pca_per_epoch(
                sample_umap_data,
                epochs=epochs,
                timepoint_col='session',
                metric_cols=['umap_1', 'umap_2']
            )
            
            assert isinstance(results, dict), "Results should be a dictionary"
            # Note: May be empty if sklearn not available
        except Exception as e:
            # Expected if sklearn not available
            assert "scikit" in str(e).lower() or "sklearn" in str(e).lower()

    def test_lasso_per_epoch_returns_dict(self, sample_umap_data):
        """LASSO per epoch should return dictionary."""
        from scripts.turning_points import lasso_per_epoch
        
        epochs = [['ses-1'], ['ses-2'], ['ses-3']]
        
        try:
            results = lasso_per_epoch(
                sample_umap_data,
                epochs=epochs,
                timepoint_col='session',
                metric_cols=['umap_1', 'umap_2'],
                group_col='group'
            )
            
            assert isinstance(results, dict), "Results should be a dictionary"
        except Exception as e:
            # Expected if sklearn not available
            assert "scikit" in str(e).lower() or "sklearn" in str(e).lower()


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
