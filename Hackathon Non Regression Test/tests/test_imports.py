"""Verifica che tutti i moduli della pipeline siano importabili."""


def test_import_packages() -> None:
    import src  # noqa: F401
    import src.data_processing  # noqa: F401
    import src.model_selection  # noqa: F401
    import src.runner  # noqa: F401
    import src.evaluation  # noqa: F401
    import src.reporting  # noqa: F401
    import src.providers  # noqa: F401
    import src.insight_agent  # noqa: F401
    import src.synthetic_dataset  # noqa: F401
    import src.prompt_optimizer  # noqa: F401
