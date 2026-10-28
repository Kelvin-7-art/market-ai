"""Random Forest optimized model descriptor."""

from market_ai.analytics.models.variant import ConfiguredVariant


class OptimizedModel(ConfiguredVariant):
    key = "optimized"
    label = "Optimized Model"