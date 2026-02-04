"""
Unit tests for create_mode_strategy factory function.

Tests the factory function that creates mode strategy instances,
including validation error paths.
"""

from unittest.mock import Mock

import pytest

from adaptive_rate_limiter.strategies.modes import (
    AccountModeStrategy,
    BasicModeStrategy,
    IntelligentModeStrategy,
    create_mode_strategy,
)


class TestCreateModeStrategySuccess:
    """Tests for successful mode strategy creation."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_retries = 3
        config.backoff_base = 2.0
        config.max_backoff = 60.0
        # Required for IntelligentModeStrategy
        config.batch_size = 50
        config.scheduler_interval = 0.001
        config.rate_limit_buffer_ratio = 0.9
        config.max_queue_size = 1000
        config.overflow_policy = "reject"
        config.max_concurrent_executions = 100
        config.request_timeout = 30.0
        return config

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        return Mock()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        return Mock()

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        return Mock()

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        return Mock()

    def test_create_basic_mode(self, mock_scheduler, mock_config, mock_client):
        """Test creating basic mode strategy."""
        strategy = create_mode_strategy(
            mode="basic",
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
        )
        assert isinstance(strategy, BasicModeStrategy)

    def test_create_basic_mode_case_insensitive(
        self, mock_scheduler, mock_config, mock_client
    ):
        """Test basic mode creation is case insensitive."""
        strategy = create_mode_strategy(
            mode="BASIC",
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
        )
        assert isinstance(strategy, BasicModeStrategy)

    def test_create_account_mode(self, mock_scheduler, mock_config, mock_client):
        """Test creating account mode strategy."""
        strategy = create_mode_strategy(
            mode="account",
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
        )
        assert isinstance(strategy, AccountModeStrategy)

    def test_create_intelligent_mode(
        self,
        mock_scheduler,
        mock_config,
        mock_client,
        mock_provider,
        mock_classifier,
        mock_state_manager,
    ):
        """Test creating intelligent mode strategy with all dependencies."""
        strategy = create_mode_strategy(
            mode="intelligent",
            scheduler=mock_scheduler,
            config=mock_config,
            client=mock_client,
            provider=mock_provider,
            classifier=mock_classifier,
            state_manager=mock_state_manager,
        )
        assert isinstance(strategy, IntelligentModeStrategy)


class TestCreateModeStrategyValidationErrors:
    """Tests for validation error paths in create_mode_strategy."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler."""
        return Mock()

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.max_retries = 3
        config.backoff_base = 2.0
        config.max_backoff = 60.0
        return config

    @pytest.fixture
    def mock_client(self):
        """Create a mock client."""
        return Mock()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        return Mock()

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock classifier."""
        return Mock()

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        return Mock()

    def test_intelligent_mode_missing_provider(
        self, mock_scheduler, mock_config, mock_client
    ):
        """Test intelligent mode fails without provider."""
        with pytest.raises(ValueError, match="ProviderInterface is required"):
            create_mode_strategy(
                mode="intelligent",
                scheduler=mock_scheduler,
                config=mock_config,
                client=mock_client,
            )

    def test_intelligent_mode_missing_classifier(
        self, mock_scheduler, mock_config, mock_client, mock_provider
    ):
        """Test intelligent mode fails without classifier."""
        with pytest.raises(ValueError, match="ClassifierProtocol is required"):
            create_mode_strategy(
                mode="intelligent",
                scheduler=mock_scheduler,
                config=mock_config,
                client=mock_client,
                provider=mock_provider,
            )

    def test_intelligent_mode_missing_state_manager(
        self, mock_scheduler, mock_config, mock_client, mock_provider, mock_classifier
    ):
        """Test intelligent mode fails without state_manager."""
        with pytest.raises(ValueError, match="StateManager is required"):
            create_mode_strategy(
                mode="intelligent",
                scheduler=mock_scheduler,
                config=mock_config,
                client=mock_client,
                provider=mock_provider,
                classifier=mock_classifier,
            )

    def test_unknown_mode(self, mock_scheduler, mock_config, mock_client):
        """Test unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler mode: bogus"):
            create_mode_strategy(
                mode="bogus",
                scheduler=mock_scheduler,
                config=mock_config,
                client=mock_client,
            )

    def test_unknown_mode_empty_string(self, mock_scheduler, mock_config, mock_client):
        """Test empty mode string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            create_mode_strategy(
                mode="",
                scheduler=mock_scheduler,
                config=mock_config,
                client=mock_client,
            )
