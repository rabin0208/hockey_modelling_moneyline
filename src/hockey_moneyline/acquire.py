"""Acquire MoneyPuck's published all-teams game-level CSV.

Failures raise AcquisitionError. This module never invents games or scores
and does not scrape unlisted MoneyPuck endpoints.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from hockey_moneyline.errors import AcquisitionError
from hockey_moneyline.provenance import SOURCE_FILE_URL, USER_AGENT

DEFAULT_CACHE_DIR = Path("data/cache/moneypuck")
CACHE_FILENAME = "all_teams.csv"


def cache_path(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / CACHE_FILENAME


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    tmp_path = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urlopen(request, timeout=120) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                raise AcquisitionError(f"Unexpected HTTP {status} from {url}")
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
    except HTTPError as exc:
        raise AcquisitionError(f"HTTP {exc.code} fetching {url}") from exc
    except URLError as exc:
        raise AcquisitionError(f"Network failure fetching {url}: {exc.reason}") from exc
    tmp_path.replace(destination)


def acquire_source_csv(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    url: str = SOURCE_FILE_URL,
) -> Path:
    path = cache_path(cache_dir)
    if path.exists() and not refresh:
        if path.stat().st_size == 0:
            raise AcquisitionError(f"Cached MoneyPuck file is empty: {path}")
        return path
    _download(url, path)
    if not path.exists() or path.stat().st_size == 0:
        raise AcquisitionError(f"Downloaded MoneyPuck file is empty: {path}")
    return path
