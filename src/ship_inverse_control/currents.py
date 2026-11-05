import functools

import xarray as xr
import numpy as np
import geopandas as gpd


@functools.cache
def create_geo_index(ds_path):
    ds = xr.open_dataset(ds_path)
    geo_array = gpd.points_from_xy(
        ds.mesh2d_face_x.values, ds.mesh2d_face_y.values, crs="EPSG:28992"
    )
    geo_index = gpd.GeoSeries(geo_array)
    return geo_index


def select_nearest_variables(
    ds, t, point, variables=("mesh2d_waterdepth", "mesh2d_ucx", "mesh2d_ucy")
):
    ds_path = ds.encoding["source"]
    geo_index = create_geo_index(ds_path)
    ds_time = ds[list(variables)].sel({"time": t}, method="nearest")
    index = geo_index.sindex.nearest(point)[1][0]
    result = ds_time.isel({"mesh2d_nFaces": index}).to_pandas()

    return result


def get_current_for_trajectory(trajectory, ds, variables=("mesh2d_ucx", "mesh2d_ucy")):
    t = trajectory.get_start_time()
    point = trajectory.get_start_location()

    result = select_nearest_variables(ds, t, point, variables=variables)
    ucx = result["mesh2d_ucx"]
    ucy = result["mesh2d_ucy"]

    V_current = np.sqrt(ucx**2 + ucy**2)
    # x and y inverted because we use nautical convention
    beta_c = np.arctan2(ucx, ucy)
    beta_current = np.rad2deg(beta_c)

    return V_current, beta_current


def get_current_for_point(point, timestamp, ds, variables=("mesh2d_ucx", "mesh2d_ucy")):
    result = select_nearest_variables(ds, timestamp, point, variables=variables)
    ucx = result["mesh2d_ucx"]
    ucy = result["mesh2d_ucy"]

    V_current = np.sqrt(ucx**2 + ucy**2)
    # x and y inverted because we use nautical convention
    beta_c = np.arctan2(ucx, ucy)
    beta_current = np.rad2deg(beta_c)

    return V_current, beta_current
