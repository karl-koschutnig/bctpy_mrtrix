"""
Test-driven development for small_worldness.py

TDD Cycle: RED -> GREEN -> REFACTOR

These tests define the expected behavior before implementation.
"""

import pytest
import numpy as np
import pandas as pd
import json
import os
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
def sample_connectome():
    """Create a sample 200x200 symmetric connectivity matrix."""
    n_nodes = 200
    # Create a random symmetric matrix with values between 0 and 1
    conn = np.random.rand(n_nodes, n_nodes)
    conn = (conn + conn.T) / 2  # Make symmetric
    np.fill_diagonal(conn, 0)  # Zero diagonal (no self-connections)
    return conn


@pytest.fixture
def sample_binary_connectome():
    """Create a sample binary connectivity matrix."""
    n_nodes = 200
    conn = np.random.rand(n_nodes, n_nodes) > 0.9  # 10% density
    conn = conn.astype(float)
    conn = (conn + conn.T) / 2
    np.fill_diagonal(conn, 0)
    return conn


@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing."""
    return pd.DataFrame({
        'participant_id': ['sub-001', 'sub-002', 'sub-003'],
        'session': ['ses-1', 'ses-2', 'ses-3'],
        'group': ['ctrl', '2w', '4w'],
        'age': [25, 30, 28],
        'sex': [0, 1, 0]
    })


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_calculate_small_worldness_accepts_non_schaefer_atlas():
    """A 166x166 (AAL3) matrix must not be rejected by a hardcoded 200-node check."""
    from scripts.small_worldness import calculate_small_worldness

    rng = np.random.default_rng(0)
    W = rng.random((166, 166))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    # Must not raise, even though 166 != the old hardcoded 200
    calculate_small_worldness(W, n_null=2, random_state=0, expected_nodes=166)


def test_calculate_small_worldness_still_validates_expected_nodes():
    """When a mismatched expected_nodes is passed, it should still be caught."""
    from scripts.small_worldness import calculate_small_worldness

    rng = np.random.default_rng(0)
    W = rng.random((166, 166))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)

    with pytest.raises(ValueError, match="200x200"):
        calculate_small_worldness(W, n_null=2, expected_nodes=200)


# ============================================================================
# TESTS: calculate_small_worldness function
# ============================================================================

class TestCalculateSmallWorldness:
    """Tests for small-worldness calculation function."""

    def test_small_worldness_returns_positive_value(self, sample_connectome):
        """Small-worldness sigma should be positive for a small-world network."""
        from scripts.small_worldness import calculate_small_worldness
        
        sigma = calculate_small_worldness(sample_connectome)
        
        assert sigma > 0, "Small-worldness should be positive"

    def test_small_worldness_lattice_network(self):
        """Lattice network should have high small-worldness."""
        from scripts.small_worldness import calculate_small_worldness
        
        n_nodes = 200
        # Create a lattice-like network (high clustering, short path length)
        conn = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(max(0, i-5), min(n_nodes, i+6)):
                if i != j:
                    conn[i, j] = 1.0
        
        sigma = calculate_small_worldness(conn)
        
        # Lattice networks typically have sigma > 1
        assert sigma > 0.5, f"Lattice network should have sigma > 0.5, got {sigma}"

    def test_small_worldness_random_network(self):
        """Random network should have sigma close to 1."""
        from scripts.small_worldness import calculate_small_worldness
        
        n_nodes = 200
        # Erdős–Rényi random network
        p = 0.1  # connection probability
        conn = np.random.rand(n_nodes, n_nodes) < p
        conn = conn.astype(float)
        conn = (conn + conn.T) / 2
        np.fill_diagonal(conn, 0)
        
        sigma = calculate_small_worldness(conn)
        
        # Random networks typically have sigma ~ 1
        assert 0.5 < sigma < 2.0, f"Random network sigma should be ~1, got {sigma}"

    def test_small_worldness_empty_network(self):
        """Empty network should have undefined/NaN small-worldness."""
        from scripts.small_worldness import calculate_small_worldness
        
        n_nodes = 200
        conn = np.zeros((n_nodes, n_nodes))
        
        sigma = calculate_small_worldness(conn)
        
        # Empty network - expect NaN or error
        assert np.isnan(sigma) or sigma == 0, "Empty network should return NaN or 0"

    def test_small_worldness_full_network(self):
        """Fully connected network should have defined small-worldness."""
        from scripts.small_worldness import calculate_small_worldness
        
        n_nodes = 200
        conn = np.ones((n_nodes, n_nodes))
        np.fill_diagonal(conn, 0)
        
        sigma = calculate_small_worldness(conn)
        
        assert not np.isnan(sigma), "Full network should have defined sigma"
        assert sigma > 0, "Full network should have positive sigma"


# ============================================================================
# TESTS: process_connectome_directory function
# ============================================================================

class TestProcessConnectomeDirectory:
    """Tests for directory processing function."""

    def test_process_directory_returns_dataframe(self, sample_connectome, temp_output_dir, mock_metadata):
        """Processing directory should return a DataFrame with sigma values."""
        from scripts.small_worldness import process_connectome_directory
        
        # Create a temporary data directory with sample connectomes
        temp_data_dir = temp_output_dir / "connectomes"
        temp_data_dir.mkdir()
        
        # Save sample connectome
        for i, (subj, ses) in enumerate(zip(
            mock_metadata['participant_id'],
            mock_metadata['session']
        )):
            conn_file = temp_data_dir / f"{subj}_{ses}.csv"
            pd.DataFrame(sample_connectome).to_csv(conn_file, index=False, header=False)
        
        # Process
        result = process_connectome_directory(
            str(temp_data_dir),
            mock_metadata,
            n_nodes=200,
            output_dir=str(temp_output_dir)
        )
        
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert 'small_worldness' in result.columns, "Result should have small_worldness column"
        assert 'participant_id' in result.columns, "Result should have participant_id column"
        assert 'session' in result.columns, "Result should have session column"

    def test_process_directory_handles_missing_files(self, sample_connectome, temp_output_dir, mock_metadata):
        """Should handle missing connectome files gracefully."""
        from scripts.small_worldness import process_connectome_directory
        
        temp_data_dir = temp_output_dir / "connectomes"
        temp_data_dir.mkdir()
        
        # Only save first connectome, skip second
        conn_file = temp_data_dir / f"{mock_metadata['participant_id'].iloc[0]}_{mock_metadata['session'].iloc[0]}.csv"
        pd.DataFrame(sample_connectome).to_csv(conn_file, index=False, header=False)
        
        # Should not raise error
        result = process_connectome_directory(
            str(temp_data_dir),
            mock_metadata,
            n_nodes=200,
            output_dir=str(temp_output_dir)
        )
        
        # Should process available files
        assert len(result) >= 1, "Should process at least one file"

    def test_process_directory_saves_output(self, sample_connectome, temp_output_dir, mock_metadata):
        """Should save results to output directory."""
        from scripts.small_worldness import process_connectome_directory
        
        temp_data_dir = temp_output_dir / "connectomes"
        temp_data_dir.mkdir()
        
        # Save sample connectomes
        for i, (subj, ses) in enumerate(zip(
            mock_metadata['participant_id'],
            mock_metadata['session']
        )):
            conn_file = temp_data_dir / f"{subj}_{ses}.csv"
            pd.DataFrame(sample_connectome).to_csv(conn_file, index=False, header=False)
        
        output_file = temp_output_dir / "small_worldness_results.csv"
        
        process_connectome_directory(
            str(temp_data_dir),
            mock_metadata,
            n_nodes=200,
            output_dir=str(temp_output_dir)
        )
        
        assert output_file.exists(), "Output file should be created"


# ============================================================================
# TESTS: CLI interface
# ============================================================================

class TestCLI:
    """Tests for command-line interface."""

    def test_main_runs_without_error(self, sample_connectome, temp_output_dir, mock_metadata, monkeypatch):
        """Main function should run without errors with proper inputs."""
        from scripts.small_worldness import main
        
        # Create temporary data
        temp_data_dir = temp_output_dir / "connectomes"
        temp_data_dir.mkdir()
        
        for i, (subj, ses) in enumerate(zip(
            mock_metadata['participant_id'],
            mock_metadata['session']
        )):
            conn_file = temp_data_dir / f"{subj}_{ses}.csv"
            pd.DataFrame(sample_connectome).to_csv(conn_file, index=False, header=False)
        
        # Save metadata
        meta_file = temp_output_dir / "metadata.csv"
        mock_metadata.to_csv(meta_file, index=False)
        
        # Mock sys.argv
        import sys
        original_argv = sys.argv
        
        try:
            sys.argv = [
                'small_worldness.py',
                '--data-dir', str(temp_data_dir),
                '--metadata-file', str(meta_file),
                '--output-dir', str(temp_output_dir),
                '--n-nodes', '200'
            ]
            
            # Should not raise
            main()
            
        finally:
            sys.argv = original_argv


# ============================================================================
# TESTS: Edge cases and error handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_symmetric_matrix_raises_error(self):
        """Non-symmetric matrix should raise ValueError."""
        from scripts.small_worldness import calculate_small_worldness
        
        conn = np.random.rand(200, 200)  # Non-symmetric
        
        with pytest.raises(ValueError, match="symmetric"):
            calculate_small_worldness(conn)

    def test_wrong_dimensions_raises_error(self):
        """Wrong matrix dimensions should raise ValueError when expected_nodes is given."""
        from scripts.small_worldness import calculate_small_worldness

        conn = np.random.rand(100, 100)  # Wrong size

        with pytest.raises(ValueError, match="200x200"):
            calculate_small_worldness(conn, expected_nodes=200)

    def test_diagonal_not_zero_warning(self):
        """Non-zero diagonal should trigger warning."""
        from scripts.small_worldness import calculate_small_worldness
        import warnings
        
        conn = np.random.rand(200, 200)
        conn = (conn + conn.T) / 2
        # Don't zero diagonal
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_small_worldness(conn)
            
            # Check if warning was raised
            assert any("diagonal" in str(warning.message).lower() for warning in w), \
                "Should warn about non-zero diagonal"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
