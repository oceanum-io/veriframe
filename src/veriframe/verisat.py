import logging
from functools import cached_property
import pandas as pd
import xarray as xr
from shapely.geometry import box, Polygon
from oceanum.datamesh import Connector

from typing import Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from veriframe.veriframe import VeriFrame
from veriframe.stats import bias, rmsd, si


logger = logging.getLogger(__name__)


from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.core.grid import RegularGrid


class VeriSat(BaseModel):
    """Base class for model verification from satellite."""

    area: Union[list, Polygon] = Field(
        description="Bounding box for verification area",
    )
    model_source: SourceFile = Field(
        description="Model data source",
    )
    model_var: str = Field(
        default="hs",
        description="Model variable to verify",
    )
    qc_level: Literal[1, 2] = Field(
        default=1,
        description="Quality control level for satellite data",
    )
    datamesh_token: Optional[str] = Field(
        default=None,
        description="Token for datamesh connector",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("area")
    @classmethod
    def to_feature(cls, v):
        if isinstance(v, list):
            return box(*v)
        return v

    @cached_property
    def datamesh(self) -> Connector:
        return Connector(token=self.datamesh_token)

    def _load_model(self, time: TimeRange) -> xr.Dataset:
        """Load the model data for the given time and grid."""
        logger.info(f"Loading the model data for {time.start} to {time.end}")
        ds = self.model_source.open()
        return ds.sel(time=slice(time.start, time.end))

    def _load_sat(self, time: TimeRange) -> pd.DataFrame:
        """Load the satellite data for the given time and grid."""
        logger.info(f"Querying satellite data for {time.start} to {time.end}")
        df = self.datamesh.query(
            datasource="imos_wave_wind",
            variables=["swh_ku_cal", "swh_ku_quality_control", "platform"],
            timefilter={"type": "range", "times": [time.start, time.end]},
            geofilter={"type": "bbox", "geom": list(self.area.bounds)},
        )
        df = df.loc[df.swh_ku_quality_control == self.qc_level]
        return df.set_index("time").sort_index()

    def get_colocs(self, time: TimeRange) -> VeriFrame:
        """Get the colocations dataframe."""
        df_sat = self._load_sat(time)
        dset_model = self._load_model(time)
        x = xr.DataArray(df_sat.longitude.values, dims=("site",))
        y = xr.DataArray(df_sat.latitude.values, dims=("site",))
        t = xr.DataArray(df_sat.index, dims=("site",))
        df_model = dset_model.interp(longitude=x, latitude=y, time=t).to_pandas()
        df = pd.concat(
            [
                df_sat.longitude,
                df_sat.latitude,
                df_sat.platform,
                df_sat.swh_ku_cal,
                df_model.hs,
            ],
            axis=1,
        )
        df.columns = ["lon", "lat", "platform", "satellite", "model"]
        return VeriFrame(df, ref_col="satellite", verify_col="model")

    def run(self, time: TimeRange) -> VeriFrame:
        """Run the verification."""
        vf = self.get_colocs(time)
        return vf
