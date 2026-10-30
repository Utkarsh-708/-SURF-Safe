import re
import ipaddress
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure", "account",
    "update", "password", "confirm", "bank", "paypal", "payment",
    "security", "wallet", "bonus", "prize", "free"
]

def has_ip_address(hostname):
    if not hostname:
        return 0
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0

def extract_features(url):
    url = str(url).strip()
    parsed = urlparse(url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    special_chars = sum(1 for c in url if c in "@?=&%_~")
    suspicious_count = sum(1 for word in SUSPICIOUS_WORDS if word in url.lower())
    subdomains = max(0, len(host.split(".")) - 2) if host else 0

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": special_chars,
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equal": url.count("="),
        "num_percent": url.count("%"),
        "num_ampersand": url.count("&"),
        "subdomain_count": subdomains,
        "has_https": int(parsed.scheme.lower() == "https"),
        "has_ip": has_ip_address(host),
        "has_port": int(parsed.port is not None) if host else 0,
        "query_length": len(query),
        "suspicious_keyword_count": suspicious_count,
        "has_suspicious_keyword": int(suspicious_count > 0),
        "has_punycode": int("xn--" in host.lower()),
    }

def extract_dataframe(urls):
    import pandas as pd
    return pd.DataFrame([extract_features(u) for u in urls])
