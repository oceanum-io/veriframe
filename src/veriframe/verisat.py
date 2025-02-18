import logging
from functools import cached_property
import pandas as pd
import xarray as xr
import geopandas as gpd
import cartopy.feature as cfeature
from shapely.geometry import box, Polygon, shape
from oceanum.datamesh import Connector

from typing import Union, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict

from veriframe.veriframe import VeriFrame


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
    extra_model_vars: list[str] = Field(
        default=[],
        description="Extra model variables to load",
    )
    sat_var: Literal["swh_ku_cal", "wspd_cal"] = Field(
        default="swh_ku_cal",
        description="Satellite variable to verify",
    )
    qc_level: Literal[1, 2] = Field(
        default=1,
        description="Quality control level for satellite data",
    )
    datamesh_token: Optional[str] = Field(
        default=None,
        description="Token for datamesh connector",
    )
    offshore_buffer: float = Field(
        default=0.5,
        description="Buffer distance for offshore mask in degrees",
    )
    scale_gshhg: Literal["110m", "50m", "10m"] = Field(
        default="110m",
        description="Scale for GSHHG land mask",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("area")
    @classmethod
    def to_feature(cls, v):
        if isinstance(v, list):
            return box(*v)
        return v

    @property
    def var(self):
        if self.model_var == "wspd":
            return "spd"
        return self.model_var

    @cached_property
    def datamesh(self) -> Connector:
        return Connector(token=self.datamesh_token)

    # @cached_property
    # def landmask(self) -> gpd.GeoDataFrame:
    #     """Load the land mask."""
    #     logger.info("Loading the land mask")
    #     gdf = self.datamesh.query(
    #         datasource="osm-land-polygons",
    #         geofilter={"type": "bbox", "geom": list(self.area.bounds)}
    #     )
    #     return gdf.dissolve()

    @cached_property
    def landmask(self):
        land = cfeature.NaturalEarthFeature(
            category="physical", name="land", scale=self.scale_gshhg
        )
        gdf = gpd.GeoDataFrame(
            geometry=[shape(feature) for feature in land.geometries()], crs="EPSG:4326"
        )
        x0, y0, x1, y1 = self.area.bounds
        return gdf.cx[x0:x1, y0:y1].dissolve()

    def _load_model(self, time: TimeRange) -> xr.Dataset:
        """Load the model data for the given time and grid."""
        logger.info(f"Loading the model data for {time.start} to {time.end}")
        model_vars = [self.model_var] + self.extra_model_vars
        ds = self.model_source.open()[model_vars]
        return ds.sel(time=slice(time.start, time.end))

    def _load_sat(self, time: TimeRange) -> pd.DataFrame:
        """Load the satellite data for the given time and grid."""
        logger.info(f"Querying satellite data for {time.start} to {time.end}")
        x0, y0, x1, y1 = list(self.area.bounds)
        df = self.datamesh.query(
            datasource="imos_wave_wind",
            variables=[self.sat_var, "swh_ku_quality_control", "platform"],
            timefilter={"type": "range", "times": [time.start, time.end]},
            geofilter={"type": "bbox", "geom": [x0%360, y0, x1%360, y1]},
        )
        # Ensure the longitude is in the range -180, 180
        df.longitude[df.longitude > 180] -= 360
        # Keep only the good data
        df = df.loc[df.swh_ku_quality_control == self.qc_level]
        return df.set_index("time").sort_index()

    def set_offshore_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the offshore mask for the dataframe."""
        gdf_points = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=self.landmask.crs,
        )
        dfout = df.copy()
        offshore = self.landmask.buffer(distance=self.offshore_buffer).iloc[0]
        dfout["is_offshore"] = gdf_points.intersects(offshore)
        dfout["offshore_distance"] = self.offshore_buffer
        return dfout

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
                df_sat[self.sat_var],
                df_model[self.model_var],
            ],
            axis=1,
        )
        df.columns = ["lon", "lat", "platform", "satellite", "model"]
        for extra_model_var in self.extra_model_vars:
            df[extra_model_var] = df_model[extra_model_var]
        df = self.set_offshore_mask(df)
        return VeriFrame(df, ref_col="satellite", verify_col="model", var=self.var)
