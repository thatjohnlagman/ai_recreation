"""
Local IP Geolocation Resolver using DB-IP City Lite MMDB.
Provides fast, offline-capable, cached IP enrichment with strict address
validation (private, documentation, reserved, loopback, and public ranges).
"""

import os
import ipaddress
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import maxminddb
except ImportError:
    maxminddb = None

GEOIP_DIR = Path(__file__).resolve().parent
DEFAULT_MMDB_PATH = GEOIP_DIR / "dbip-city-lite.mmdb"

# Documentation and special-purpose test ranges (RFC 5737, RFC 6598)
DOCUMENTATION_NETWORKS = [
    ipaddress.ip_network("192.0.2.0/24"),      # TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),   # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),    # TEST-NET-3
    ipaddress.ip_network("100.64.0.0/10"),     # Shared Address Space
    ipaddress.ip_network("240.0.0.0/4"),       # Reserved / Future use
]

DB_ATTRIBUTION_TEXT = "IP Geolocation by DB-IP (https://db-ip.com)"
DB_ATTRIBUTION_HTML = '<a href="https://db-ip.com" target="_blank" rel="noopener">IP Geolocation by DB-IP</a>'

_READER_INSTANCE = None
_READER_INITIALIZED = False


def get_mmdb_reader():
    """Lazily initializes and caches the MMDB reader instance."""
    global _READER_INSTANCE, _READER_INITIALIZED
    if _READER_INITIALIZED:
        return _READER_INSTANCE

    _READER_INITIALIZED = True
    if maxminddb is None:
        print("[GeoIP] Warning: maxminddb library not installed. Geolocation disabled.")
        return None

    db_path = Path(os.environ.get("IDS_GEOIP_DB_PATH", str(DEFAULT_MMDB_PATH)))
    if not db_path.exists():
        print(f"[GeoIP] Warning: MMDB database not found at {db_path}. Geolocation disabled.")
        return None

    try:
        _READER_INSTANCE = maxminddb.open_database(str(db_path))
        print(f"[GeoIP] Successfully loaded MMDB database from {db_path}.")
    except Exception as e:
        print(f"[GeoIP] Failed to open MMDB database at {db_path}: {e}")
        _READER_INSTANCE = None

    return _READER_INSTANCE


@lru_cache(maxsize=4096)
def resolve_ip_geo(ip_str: Optional[str]) -> Dict[str, Any]:
    """
    Resolves geolocation information for an IP address.
    Returns structured dictionary:
      - status: 'geolocated', 'private', 'documentation', 'loopback', 'reserved', 'unknown', 'invalid', 'db_unavailable'
      - country: str or '—'
      - city: str or '—'
      - location_str: human-readable location display
      - lat: float or None
      - lng: float or None
    """
    if not ip_str or not isinstance(ip_str, str):
        return {
            "status": "invalid",
            "country": "—",
            "city": "—",
            "location_str": "—",
            "lat": None,
            "lng": None,
        }

    ip_str = ip_str.strip()

    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return {
            "status": "invalid",
            "country": "—",
            "city": "—",
            "location_str": "Invalid Address",
            "lat": None,
            "lng": None,
        }

    # 1. Documentation / Test Networks (e.g., TEST-NET-3: 203.0.113.x, TEST-NET-2: 198.51.100.x)
    for net in DOCUMENTATION_NETWORKS:
        if ip_obj in net:
            # Preserves truthful 'Unknown' display and no coordinates for documentation/test IPs
            return {
                "status": "documentation",
                "country": "—",
                "city": "—",
                "location_str": "Unknown",
                "lat": None,
                "lng": None,
            }

    # 2. RFC 1918 Private subnets & Local addresses
    if ip_obj.is_private:
        return {
            "status": "private",
            "country": "—",
            "city": "—",
            "location_str": "Private Network",
            "lat": None,
            "lng": None,
        }

    # 7. Routable Public IP -> Query MMDB
    reader = get_mmdb_reader()
    if reader is None:
        return {
            "status": "db_unavailable",
            "country": "—",
            "city": "—",
            "location_str": "Unknown",
            "lat": None,
            "lng": None,
        }

    try:
        record = reader.get(ip_str)
        if not record:
            return {
                "status": "unknown",
                "country": "—",
                "city": "—",
                "location_str": "Unknown",
                "lat": None,
                "lng": None,
            }

        country_dict = record.get("country", {})
        country_name = country_dict.get("names", {}).get("en") or country_dict.get("iso_code")
        city_name = record.get("city", {}).get("names", {}).get("en")
        loc_dict = record.get("location", {})

        raw_lat = loc_dict.get("latitude")
        raw_lng = loc_dict.get("longitude")

        lat = float(raw_lat) if raw_lat is not None else None
        lng = float(raw_lng) if raw_lng is not None else None

        has_valid_coords = (
            lat is not None
            and lng is not None
            and (-90.0 <= lat <= 90.0)
            and (-180.0 <= lng <= 180.0)
        )

        if has_valid_coords:
            if city_name and country_name:
                loc_str = f"{city_name}, {country_name}"
            elif country_name:
                loc_str = country_name
            else:
                loc_str = f"{lat:.2f}, {lng:.2f}"

            return {
                "status": "geolocated",
                "country": country_name or "—",
                "city": city_name or "—",
                "location_str": loc_str,
                "lat": lat,
                "lng": lng,
            }
        elif country_name:
            return {
                "status": "country_only",
                "country": country_name,
                "city": city_name or "—",
                "location_str": country_name,
                "lat": None,
                "lng": None,
            }
        else:
            return {
                "status": "unknown",
                "country": "—",
                "city": "—",
                "location_str": "Unknown",
                "lat": None,
                "lng": None,
            }
    except Exception as e:
        return {
            "status": "error",
            "country": "—",
            "city": "—",
            "location_str": "Unknown",
            "lat": None,
            "lng": None,
        }
