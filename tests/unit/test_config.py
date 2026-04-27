"""Unit tests for pysparta.config module.

Tests cover:
- Configuration file initialization
- Reading and writing configuration options
- get_option and set_option functions
- Path handling for data directories
"""

from pathlib import Path
from unittest.mock import patch
import tomlkit

from pysparta import config


class TestConfigFileOperations:
    """Test suite for configuration file operations."""

    @patch('pysparta.config.platformdirs.user_config_path')
    def test_get_config_path(self, mock_user_config_path, tmp_path):
        """Test that config path is correctly retrieved."""
        mock_user_config_path.return_value = tmp_path
        path = config.get_config_path()
        assert path == tmp_path / "config.toml"
        assert path.name == "config.toml"

    @patch('pysparta.config.get_config_path')
    def test_init_config_file_creates_file(self, mock_get_path, tmp_path):
        """Test that initialization creates the config file."""
        config_file = tmp_path / "config.toml"
        mock_get_path.return_value = config_file
        
        config._init_config_file()
        
        assert config_file.exists()
        content = config_file.read_text()
        assert "[crs_soda]" in content
        assert "[merra2_daily]" in content
        assert "[sunwhere]" in content

    @patch('pysparta.config.get_config_path')
    def test_read_config_options_existing_file(self, mock_get_path, tmp_path):
        """Test reading from an existing config file."""
        config_file = tmp_path / "config.toml"
        test_config = """
[test_table]
test_option = "test_value"
"""
        config_file.write_text(test_config)
        mock_get_path.return_value = config_file
        
        options = config._read_config_options()
        
        assert "test_table" in options
        assert options["test_table"]["test_option"] == "test_value"


class TestConfigOptions:
    """Test suite for get_option and set_option functions."""

    def test_get_option_existing(self):
        """Test getting an existing option."""
        # Mock the global config
        with patch.object(config, '_GLOBAL_CONFIG', {
            'test_table': {'test_key': 'test_value'}
        }):
            result = config.get_option('test_table.test_key')
            assert result == 'test_value'

    def test_get_option_missing_table(self):
        """Test getting option from non-existent table."""
        with patch.object(config, '_GLOBAL_CONFIG', {}):
            result = config.get_option('missing_table.test_key')
            assert result is None

    def test_get_option_missing_key(self):
        """Test getting non-existent option from existing table."""
        with patch.object(config, '_GLOBAL_CONFIG', {
            'test_table': {}
        }):
            result = config.get_option('test_table.missing_key')
            assert result is None

    def test_get_option_with_default(self):
        """Test getting option with default value."""
        with patch.object(config, '_GLOBAL_CONFIG', {
            'test_table': {}
        }):
            result = config.get_option('test_table.missing_key', default='default_value')
            assert result == 'default_value'

    def test_get_option_data_dir_returns_path(self):
        """Test that data_dir options return Path objects."""
        with patch.object(config, '_GLOBAL_CONFIG', {
            'merra2_daily': {'data_dir': '/some/path'}
        }):
            result = config.get_option('merra2_daily.data_dir')
            assert isinstance(result, Path)
            assert result == Path('/some/path')

    def test_set_option_existing_table(self):
        """Test setting an option in an existing table."""
        test_config = {'test_table': {'test_key': 'old_value'}}
        with patch.object(config, '_GLOBAL_CONFIG', test_config):
            config.set_option('test_table.test_key', 'new_value')
            assert config._GLOBAL_CONFIG['test_table']['test_key'] == 'new_value'

    def test_set_option_missing_table(self):
        """Test setting option in non-existent table."""
        with patch.object(config, '_GLOBAL_CONFIG', {}):
            result = config.set_option('missing_table.test_key', 'value')
            assert result is None

    def test_set_option_data_dir_with_path(self):
        """Test that Path objects are converted to strings for data_dir."""
        test_config = {'test_table': {}}
        with patch.object(config, '_GLOBAL_CONFIG', test_config):
            config.set_option('test_table.data_dir', Path('/some/path'))
            assert config._GLOBAL_CONFIG['test_table']['data_dir'] == '/some/path'

    def test_set_option_creates_new_key(self):
        """Test that set_option creates new keys in existing tables."""
        test_config = {'test_table': {}}
        with patch.object(config, '_GLOBAL_CONFIG', test_config):
            config.set_option('test_table.new_key', 'new_value')
            assert config._GLOBAL_CONFIG['test_table']['new_key'] == 'new_value'


class TestConfigReset:
    """Test suite for config reset functionality."""

    @patch('pysparta.config.get_config_path')
    @patch('pysparta.config._read_config_options')
    def test_reset_config_file(self, mock_read, mock_get_path, tmp_path):
        """Test that reset removes the file and reloads config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[test]\nkey = 'value'")
        mock_get_path.return_value = config_file
        mock_read.return_value = {}
        
        config._reset_config_file()
        
        assert not config_file.exists()
        mock_read.assert_called_once()


class TestShowConfig:
    """Test suite for show_config function."""

    def test_show_config_returns_none(self):
        """Test that show_config executes without errors."""
        with patch.object(config, '_GLOBAL_CONFIG', {'test': {'key': 'value'}}):
            # Should not raise any exception
            result = config.show_config()
            # pprint returns None
            assert result is None


class TestDefaultConfigStructure:
    """Test suite to verify the default configuration structure."""

    def test_default_config_has_required_sections(self):
        """Test that default config contains all required sections."""
        default_config = tomlkit.loads(config._DEFAULT_CONFIG_TOML_)
        
        assert 'crs_soda' in default_config
        assert 'merra2_gee' in default_config
        assert 'merra2_daily' in default_config
        assert 'sunwhere' in default_config

    def test_sunwhere_default_algorithm(self):
        """Test that sunwhere section has default algorithm."""
        default_config = tomlkit.loads(config._DEFAULT_CONFIG_TOML_)
        
        assert default_config['sunwhere']['algorithm'] == 'psa'
        assert default_config['sunwhere']['refraction'] is True
        assert default_config['sunwhere']['engine'] == 'numexpr'


class TestConfigIntegration:
    """Integration tests for configuration workflow."""

    @patch('pysparta.config.get_config_path')
    def test_full_config_workflow(self, mock_get_path, tmp_path):
        """Test a complete workflow: init, read, set, get."""
        config_file = tmp_path / "config.toml"
        mock_get_path.return_value = config_file
        
        # Initialize config
        config._init_config_file()
        assert config_file.exists()
        
        # Read config
        options = config._read_config_options()
        assert 'sunwhere' in options
        
        # Simulate setting and getting with the loaded config
        with patch.object(config, '_GLOBAL_CONFIG', options):
            # Get existing option
            algo = config.get_option('sunwhere.algorithm')
            assert algo == 'psa'
            
            # Set new value
            config.set_option('sunwhere.algorithm', 'spa')
            new_algo = config.get_option('sunwhere.algorithm')
            assert new_algo == 'spa'
