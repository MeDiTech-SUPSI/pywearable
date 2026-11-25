from pywearable.loader.base import BaseLoader


class LoaderBase(BaseLoader):
    """Minimal BaseLoader implementation for focused domain tests."""

    def load_sleep_summaries(self, *args, **kwargs):
        raise NotImplementedError

    def load_sleep_stages(self, *args, **kwargs):
        raise NotImplementedError

    def load_heart_rate(self, *args, **kwargs):
        raise NotImplementedError
