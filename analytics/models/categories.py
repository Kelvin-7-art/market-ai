"""Sidebar configuration for Analytics model families and variants."""

MODEL_CATEGORIES: dict[str, dict[str, object]] = {
    "Random Forest": {"icon": "🌲", "slug": "random_forest", "variants": ["Base Model", "Advanced Model", "Optimized Model", "Fast Model", "With SMOTE"]},
    "XGBoost": {"icon": "⚡", "slug": "xgboost", "variants": ["Base Model", "Advanced Model", "Optimized Model"]},
    "LightGBM": {"icon": "💡", "slug": "lightgbm", "variants": ["Base Model", "Advanced Model", "Optimized Model"]},
    "SVM": {"icon": "🎯", "slug": "svm", "variants": ["Base Model", "Tuned Model", "Optimized Model"]},
    "Neural Network": {"icon": "🧠", "slug": "neural_network", "variants": ["Base Model", "Deep Model", "Optimized Model"]},
    "LSTM": {"icon": "📊", "slug": "lstm", "variants": ["Base Model", "Tuned Model", "Optimized Model"]},
    "Transformer": {"icon": "🔮", "slug": "transformer", "variants": ["Base Model", "Fine-tuned Model", "Optimized Model"]},
}


def variants_for(category: str) -> list[str]:
    """Return configured variants for a model family."""
    config = MODEL_CATEGORIES.get(category, MODEL_CATEGORIES["Random Forest"])
    return list(config["variants"])