"""Random Forest SMOTE model descriptor."""

from market_ai.analytics.models.variant import ConfiguredVariant


class SmoteModel(ConfiguredVariant):
    key = "smote"
    label = "With SMOTE"