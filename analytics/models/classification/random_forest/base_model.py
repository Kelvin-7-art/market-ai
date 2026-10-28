"""Random Forest base model descriptor."""

from market_ai.analytics.models.variant import ConfiguredVariant


class BaseModel(ConfiguredVariant):
    key = "base"
    label = "Base Model"