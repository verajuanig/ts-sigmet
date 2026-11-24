#Vamos a recortar los datos full disk de GOES a nuestra region

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime
from datetime import datetime, timedelta


from constantes import BOUNDS
#BOUNDS = {
#    "data_lat_min": -47.0,
#    "data_lat_max": -18.5,
#    "data_lon_min": -75.5,
#    "data_lon_max": -37.5,
#}

# Caché en memoria para no recalcular la proyección fija ABI en cada timestep
_ABI_GRID_CACHE: Dict[Tuple, Dict[str, np.ndarray | slice]] = {}


def _abi_angles_to_latlon(
    x: np.ndarray, y: np.ndarray, proj_var: nc.Variable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convierte los ángulos de barrido fijo ABI (radianes) a latitud y longitud geodésicas (grados)
    según el PUG (Product Definition and Users' Guide) de GOES-R.
    """
    r_eq = float(proj_var.semi_major_axis)
    r_pol = float(proj_var.semi_minor_axis)
    h = float(proj_var.perspective_point_height) + r_eq
    lon_0 = np.deg2rad(float(proj_var.longitude_of_projection_origin))

    xx, yy = np.meshgrid(x, y)

    sin_x, cos_x = np.sin(xx), np.cos(xx)
    sin_y, cos_y = np.sin(yy), np.cos(yy)

    a = sin_x**2 + (cos_x**2) * (cos_y**2 + ((r_eq / r_pol) ** 2) * (sin_y**2))
    b = -2.0 * h * cos_x * cos_y
    c = (h**2) - (r_eq**2)

    discriminant = (b**2) - 4.0 * a * c
    # Evitar advertencias fuera del disco terrestre
    valid = discriminant >= 0
    r_s = np.full_like(xx, np.nan, dtype=np.float64)
    r_s[valid] = (-b[valid] - np.sqrt(discriminant[valid])) / (2.0 * a[valid])

    s_x = r_s * cos_x * cos_y
    s_y = -r_s * sin_x
    s_z = r_s * cos_x * sin_y

    lat = np.rad2deg(
        np.arctan(((r_eq / r_pol) ** 2) * (s_z / np.sqrt(((h - s_x) ** 2) + (s_y**2))))
    )
    lon = np.rad2deg(lon_0 - np.arctan2(s_y, h - s_x))

    return lat.astype(np.float32), lon.astype(np.float32)


def get_abi_subset_indices(
    ds: nc.Dataset, bounds: Dict[str, float] = BOUNDS
) -> Dict[str, np.ndarray | slice]:
    """
    Obtiene y cachea los slices (y, x) y las grillas 2D de lat/lon recortadas
    para un archivo ABI dado.
    """
    proj = ds.variables["goes_imager_projection"]
    lon_0 = float(proj.longitude_of_projection_origin)
    shape = (ds.dimensions["y"].size, ds.dimensions["x"].size)

    cache_key = (
        lon_0,
        shape,
        bounds["data_lat_min"],
        bounds["data_lat_max"],
        bounds["data_lon_min"],
        bounds["data_lon_max"],
    )

    if cache_key in _ABI_GRID_CACHE:
        return _ABI_GRID_CACHE[cache_key]

    x = ds.variables["x"][:].filled(np.nan)
    y = ds.variables["y"][:].filled(np.nan)
    lat_full, lon_full = _abi_angles_to_latlon(x, y, proj)

    mask = (
        (lat_full >= bounds["data_lat_min"])
        & (lat_full <= bounds["data_lat_max"])
        & (lon_full >= bounds["data_lon_min"])
        & (lon_full <= bounds["data_lon_max"])
    )

    y_idx, x_idx = np.where(mask)
    if len(y_idx) == 0 or len(x_idx) == 0:
        raise ValueError("El dominio definido en BOUNDS cae completamente fuera del disco ABI.")

    y_slice = slice(int(y_idx.min()), int(y_idx.max()) + 1)
    x_slice = slice(int(x_idx.min()), int(x_idx.max()) + 1)

    grid_info = {
        "y_slice": y_slice,
        "x_slice": x_slice,
        "lat": lat_full[y_slice, x_slice],
        "lon": lon_full[y_slice, x_slice],
    }
    _ABI_GRID_CACHE[cache_key] = grid_info
    return grid_info


def read_abi_cropped(
    filepath: str | Path,
    variable_name: str,
    bounds: Dict[str, float] = BOUNDS,
    apply_dqf: bool = True,
) -> Dict[str, np.ndarray | str]:
    """
    Lee únicamente la ventana espacial recortada de un producto ABI (ej. 'CMI' para C13
    o 'PRES' para CTPF).
    """
    filepath = Path(filepath)
    with nc.Dataset(filepath, mode="r") as ds:
        grid = get_abi_subset_indices(ds, bounds=bounds)
        ys, xs = grid["y_slice"], grid["x_slice"]

        var = ds.variables[variable_name]
        # netCDF4 aplica automáticamente scale_factor, add_offset y convierte _FillValue en MaskedArray
        data_ma = var[ys, xs]
        data = np.ma.filled(data_ma, np.nan).astype(np.float32)

        # Filtrar píxeles de mala calidad usando Data Quality Flags (DQF == 0 es buena calidad)
        if apply_dqf and "DQF" in ds.variables:
            dqf = np.ma.filled(ds.variables["DQF"][ys, xs], 255)
            data[dqf != 0] = np.nan

        return {
            "data": data,
            "lat": grid["lat"],
            "lon": grid["lon"],
            "units": getattr(var, "units", ""),
            "time_coverage_start": getattr(ds, "time_coverage_start", ""),
            "time_coverage_end": getattr(ds, "time_coverage_end", ""),
        }


def read_glm_window_cropped(
    glm_files: List[str | Path],
    bounds: Dict[str, float] = BOUNDS,
    min_quality_flag: int = 0,
) -> pd.DataFrame:
    """
    Combina una lista de archivos GLM-L2-LCFA (20s cada uno) y filtra los flashes
    dentro de BOUNDS en un único DataFrame listo para análisis o ploteo.
    """
    records = []

    for fpath in glm_files:
        fpath = Path(fpath)
        if not fpath.exists():
            continue

        with nc.Dataset(fpath, mode="r") as ds:
            if "flash_lat" not in ds.variables or ds.dimensions["number_of_flashes"].size == 0:
                continue

            lat = np.ma.filled(ds.variables["flash_lat"][:], np.nan)
            lon = np.ma.filled(ds.variables["flash_lon"][:], np.nan)
            qflag = np.ma.filled(ds.variables["flash_quality_flag"][:], -1)

            mask = (
                (lat >= bounds["data_lat_min"])
                & (lat <= bounds["data_lat_max"])
                & (lon >= bounds["data_lon_min"])
                & (lon <= bounds["data_lon_max"])
                & (qflag == min_quality_flag)
            )

            if not np.any(mask):
                continue

            # Extraer tiempo base del producto para reconstruir el timestamp exacto de cada flash
            prod_time_var = ds.variables["product_time"]
            base_time = nc.num2date(
                prod_time_var[0],
                units=prod_time_var.units,
                only_use_cftime_datetimes=False,
            )
            offset_sec = np.ma.filled(
                ds.variables["flash_time_offset_of_first_event"][mask], 0.0
            )
            flash_times = pd.to_datetime(base_time, utc=True) + pd.to_timedelta(
                offset_sec, unit="s"
            )

            df_chunk = pd.DataFrame(
                {
                    "time_utc": flash_times,
                    "flash_id": np.ma.filled(ds.variables["flash_id"][mask], -1),
                    "lat": lat[mask].astype(np.float32),
                    "lon": lon[mask].astype(np.float32),
                    "energy_j": np.ma.filled(ds.variables["flash_energy"][mask], np.nan).astype(np.float32),
                    "area_km2": np.ma.filled(ds.variables["flash_area"][mask], np.nan).astype(np.float32),
                }
            )
            records.append(df_chunk)

    if not records:
        return pd.DataFrame(
            columns=["time_utc", "flash_id", "lat", "lon", "energy_j", "area_km2"]
        )

    df_all = pd.concat(records, ignore_index=True)
    # Eliminar posibles duplicados si hubiera solapamiento entre archivos
    df_all = df_all.drop_duplicates(subset=["flash_id"]).sort_values("time_utc").reset_index(drop=True)
    return df_all


def load_goes_window_cropped(
    bundle: Dict[str, List[Path] | Optional[Path]],
    bounds: Dict[str, float] = BOUNDS,
) -> Dict[str, object]:
    """
    Punto de entrada unificado: toma el diccionario devuelto por `download_goes_bundle`
    y devuelve C13, CTPF y GLM recortados al dominio espacial.
    """
    output = {
        "c13": None,
        "ctpf": None,
        "glm_flashes": pd.DataFrame(),
    }

    if bundle.get("c13"):
        # En ABI-L2-CMIPF la variable es 'CMI' (Brightness Temperature en Kelvin)
        c13 = read_abi_cropped(bundle["c13"], variable_name="CMI", bounds=bounds)
        c13["data_celsius"] = c13["data"] - 273.15
        output["c13"] = c13

    if bundle.get("ctpf"):
        # En ABI-L2-CTPF la variable es 'PRES' (Cloud Top Pressure en hPa)
        output["ctpf"] = read_abi_cropped(bundle["ctpf"], variable_name="PRES", bounds=bounds)

    if bundle.get("glm"):
        output["glm_flashes"] = read_glm_window_cropped(bundle["glm"], bounds=bounds)

    return output



from datetime import datetime, timezone
from goes_download import download_goes_bundle
#from goes_reader import load_goes_window_cropped, BOUNDS

dt_evento = datetime(2026, 11, 15, 18, 30, tzinfo=timezone.utc)

# 1. Descargar C13, CTPF y los 30 archivos GLM (ventana de 10 minutos)
bundle = download_goes_bundle(
    target_time=dt_evento,
    output_dir="../datos_goes",
    satellite="goes19",
    glm_window_minutes=10
)

# 2. Levantar y recortar espacialmente al dominio Sudamérica / Argentina central
datos = load_goes_window_cropped(bundle, bounds=BOUNDS)

print("Forma C13 recortado:", datos["c13"]["data"].shape)
print("Forma CTPF recortado:", datos["ctpf"]["data"].shape)
print("Total flashes GLM en el dominio:", len(datos["glm_flashes"]))
print(datos["glm_flashes"].head())