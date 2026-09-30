import pandas as pd
import pytest

from pywearable.loader.base import BaseLoader, resolve_user_ids


class MinimalLoader(BaseLoader):
    def load_sleep_summary(self, *args, **kwargs):
        return pd.DataFrame()

    def load_sleep_stage(self, *args, **kwargs):
        return pd.DataFrame()

    def load_heart_rate(self, *args, **kwargs):
        return pd.DataFrame()

    def get_user_ids(self):
        return ["user-a", "user-b"]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("all", ["user-a", "user-b"]),
        (None, ["user-a", "user-b"]),
        ("user-a", ["user-a"]),
        (("user-b", "user-a"), ["user-b", "user-a"]),
    ],
)
def test_resolve_user_ids(requested, expected):
    loader = MinimalLoader()

    assert resolve_user_ids(loader, requested) == expected
    assert loader.resolve_user_ids(requested) == expected
