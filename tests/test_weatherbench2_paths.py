import importlib
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from geoxplain_aurora_adapter.serving.config import (
    PUBLIC_WEATHERBENCH2_BUCKET,
    PUBLIC_WEATHERBENCH2_ERA5_PATH,
    normalize_weatherbench2_path,
)


class WeatherBench2PathTests(unittest.TestCase):
    def _import_data_with_fakes(self, *, open_zarr=None):
        fake_xarray = types.SimpleNamespace(open_zarr=open_zarr or Mock())
        fake_pandas = types.SimpleNamespace(DatetimeIndex=lambda values: values)

        with patch.dict(sys.modules, {"xarray": fake_xarray, "pandas": fake_pandas}):
            sys.modules.pop("geoxplain_aurora_adapter.engine.data", None)
            data = importlib.import_module("geoxplain_aurora_adapter.engine.data")
        self.addCleanup(sys.modules.pop, "geoxplain_aurora_adapter.engine.data", None)
        return data, fake_xarray

    def test_public_bucket_alias_resolves_to_era5_zarr_store(self):
        self.assertEqual(
            normalize_weatherbench2_path(PUBLIC_WEATHERBENCH2_BUCKET),
            PUBLIC_WEATHERBENCH2_ERA5_PATH,
        )
        self.assertEqual(
            normalize_weatherbench2_path(f"{PUBLIC_WEATHERBENCH2_BUCKET}/era5/"),
            PUBLIC_WEATHERBENCH2_ERA5_PATH,
        )

    def test_data_loader_preserves_gs_uri_from_environment(self):
        with patch.dict(os.environ, {"GEOXPLAIN_AURORA_ADAPTER_WB2_PATHS": PUBLIC_WEATHERBENCH2_BUCKET}):
            data, _ = self._import_data_with_fakes()

        self.assertEqual(data.WB2_PATHS, (PUBLIC_WEATHERBENCH2_ERA5_PATH,))

    def test_data_loader_opens_gs_path_without_local_exists_check(self):
        fake_ds = types.SimpleNamespace(time=types.SimpleNamespace(values=["2020-01-01T00:00:00"]))
        data, fake_xarray = self._import_data_with_fakes(open_zarr=Mock(return_value=fake_ds))

        data.WB2_PATHS = (PUBLIC_WEATHERBENCH2_BUCKET,)
        data._WB2_DATASET_CACHE.clear()

        with patch("geoxplain_aurora_adapter.engine.data.os.path.exists", return_value=False) as exists:
            self.assertIs(data._find_store("2020-01-01T00:00:00"), fake_ds)

        exists.assert_not_called()
        fake_xarray.open_zarr.assert_called_once_with(
            PUBLIC_WEATHERBENCH2_ERA5_PATH,
            storage_options={"token": "anon"},
        )


if __name__ == "__main__":
    unittest.main()
