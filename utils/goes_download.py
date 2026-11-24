import os
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import s3fs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Buckets públicos oficiales en AWS (no requieren credenciales)
GOES_BUCKETS = {
    "goes16": "noaa-goes16",
    "goes19": "noaa-goes19",
}

# Regex para extraer el timestamp de inicio (sYYYYJJJHHMMSSs) del nombre de archivo GOES
START_TIME_REGEX = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def get_s3_client() -> s3fs.S3FileSystem:
    """Inicializa un cliente S3 anónimo para los buckets públicos de NOAA."""
    return s3fs.S3FileSystem(anon=True)


def _parse_goes_start_time(filepath: str) -> Optional[datetime]:
    """Extrae el datetime UTC de inicio de escaneo desde el nombre del archivo NetCDF."""
    match = START_TIME_REGEX.search(os.path.basename(filepath))
    if not match:
        return None
    year, doy, hour, minute, second = map(int, match.groups())
    dt = datetime(year, 1, 1, hour, minute, second, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    return dt


def find_glm_files(
    fs: s3fs.S3FileSystem,
    start_time: datetime,
    bucket_name: str,
    minutes: int = 10
) -> List[str]:
    """
    Lista los archivos GLM-L2-LCFA en [start_time, start_time + minutes)
    haciendo un único listado por hora en S3.
    """
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = start_time + timedelta(minutes=minutes)

    # Identificar todas las horas UTC que abarca la ventana (por si cruza cambio de hora)
    hours_to_check = set()
    cursor = start_time.replace(minute=0, second=0, microsecond=0)
    while cursor <= end_time:
        hours_to_check.add(cursor)
        cursor += timedelta(hours=1)

    matched_files = []
    for hr_dt in sorted(hours_to_check):
        folder_path = f"{bucket_name}/GLM-L2-LCFA/{hr_dt.strftime('%Y/%j/%H/')}"
        try:
            files_in_hour = fs.ls(folder_path, detail=False)
        except FileNotFoundError:
            logger.warning(f"No existe la carpeta GLM en S3: {folder_path}")
            continue

        for f in files_in_hour:
            if not f.endswith(".nc"):
                continue
            file_dt = _parse_goes_start_time(f)
            if file_dt and (start_time <= file_dt < end_time):
                matched_files.append(f)

    return sorted(matched_files)


def find_closest_abi_file(
    fs: s3fs.S3FileSystem,
    target_time: datetime,
    bucket_name: str,
    product: str,
    channel: Optional[str] = None,
    tolerance_minutes: int = 6
) -> Optional[str]:
    """
    Busca el archivo ABI (ej. ABI-L2-CMIPF o ABI-L2-CTPF) más cercano a target_time.
    """
    if target_time.tzinfo is None:
        target_time = target_time.replace(tzinfo=timezone.utc)

    # Revisar la hora actual y adyacentes por si target_time está cerca del minuto :00 o :59
    candidate_hours = {
        (target_time - timedelta(minutes=tolerance_minutes)).replace(minute=0, second=0, microsecond=0),
        target_time.replace(minute=0, second=0, microsecond=0),
        (target_time + timedelta(minutes=tolerance_minutes)).replace(minute=0, second=0, microsecond=0),
    }

    candidates = []
    for hr_dt in candidate_hours:
        folder = f"{bucket_name}/{product}/{hr_dt.strftime('%Y/%j/%H/')}"
        try:
            files = fs.ls(folder, detail=False)
        except FileNotFoundError:
            continue

        for f in files:
            if not f.endswith(".nc"):
                continue
            if channel and f"-M6{channel}_" not in f and f"-M3{channel}_" not in f and f"-M4{channel}_" not in f:
                continue
            file_dt = _parse_goes_start_time(f)
            if file_dt:
                diff = abs((file_dt - target_time).total_seconds())
                if diff <= tolerance_minutes * 60:
                    candidates.append((diff, f))

    if not candidates:
        return None

    # Devolver el archivo con menor diferencia temporal
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _download_single_file(
    fs: s3fs.S3FileSystem,
    s3_path: str,
    local_dir: Path,
    overwrite: bool = False
) -> Path:
    """Descarga un archivo de S3 a local_dir preservando el nombre original."""
    local_path = local_dir / os.path.basename(s3_path)
    if local_path.exists() and not overwrite:
        logger.debug(f"Ya existe localmente (omitido): {local_path.name}")
        return local_path

    local_dir.mkdir(parents=True, exist_ok=True)
    # Descarga atómica usando archivo temporal .part para evitar NetCDFs corruptos
    tmp_path = local_path.with_suffix(".nc.part")
    try:
        fs.get(s3_path, str(tmp_path))
        tmp_path.replace(local_path)
        logger.info(f"Descargado: {local_path.name}")
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Error descargando {s3_path}: {e}") from e

    return local_path


def download_goes_bundle(
    target_time: datetime,
    output_dir: str | Path,
    satellite: str = "goes19",
    glm_window_minutes: int = 10,
    max_workers: int = 8,
    overwrite: bool = False
) -> Dict[str, List[Path] | Optional[Path]]:
    """
    Descarga en paralelo C13, CTPF y todos los archivos GLM de la ventana asociada.
    """
    bucket = GOES_BUCKETS.get(satellite.lower(), satellite)
    out_path = Path(output_dir)
    fs = get_s3_client()

    logger.info(f"Buscando productos en {bucket} alrededor de {target_time.isoformat()} UTC...")

    c13_s3 = find_closest_abi_file(fs, target_time, bucket, product="ABI-L2-CMIPF", channel="C13")
    ctpf_s3 = find_closest_abi_file(fs, target_time, bucket, product="ABI-L2-CTPF")
    glm_s3_list = find_glm_files(fs, target_time, bucket, minutes=glm_window_minutes)

    results: Dict[str, List[Path] | Optional[Path]] = {
        "c13": None,
        "ctpf": None,
        "glm": []
    }

    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        if c13_s3:
            tasks.append((
                "c13",
                executor.submit(_download_single_file, fs, c13_s3, out_path / "ABI-L2-CMIPF-C13", overwrite)
            ))
        else:
            logger.warning("No se encontró archivo ABI-L2-CMIPF C13 para el timestamp dado.")

        if ctpf_s3:
            tasks.append((
                "ctpf",
                executor.submit(_download_single_file, fs, ctpf_s3, out_path / "ABI-L2-CTPF", overwrite)
            ))
        else:
            logger.warning("No se encontró archivo ABI-L2-CTPF para el timestamp dado.")

        for glm_file in glm_s3_list:
            tasks.append((
                "glm",
                executor.submit(_download_single_file, fs, glm_file, out_path / "GLM-L2-LCFA", overwrite)
            ))

        for category, future in tasks:
            downloaded_path = future.result()
            if category == "glm":
                results["glm"].append(downloaded_path)
            else:
                results[category] = downloaded_path

    results["glm"].sort()
    logger.info(
        f"Descarga finalizada -> C13: {bool(results['c13'])}, "
        f"CTPF: {bool(results['ctpf'])}, GLM: {len(results['glm'])} archivos."
    )
    return results


#Esto para la descarga de datos en full disk
if __name__ == "__main__":
    # Ejemplo de uso
    dt_evento = datetime(2026, 11, 20, 18, 50, tzinfo=timezone.utc)
    archivos = download_goes_bundle(
        target_time=dt_evento,
        output_dir="../datos_goes",
        satellite="goes19",
        glm_window_minutes=10
    )


