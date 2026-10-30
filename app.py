import os
import sys
import html
import difflib
import json
import urllib.request
import urllib.error
import socket
import ssl
import ipaddress
from urllib.parse import urlsplit, urljoin, urlencode
from datetime import datetime
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, "src"))

from feature_extraction import extract_features
from predict import explain

MODEL_PATH = os.path.join(BASE, "models", "phishing_model.pkl")

# Scan history is stored locally in the project's results folder.
RESULT_DIR = os.path.join(BASE, "results")
HISTORY_PATH = os.path.join(RESULT_DIR, "scan_history.csv")
os.makedirs(RESULT_DIR, exist_ok=True)


def save_scan_history(url, prediction, probability):
    """Append one successful scan to the local history CSV."""
    record = pd.DataFrame([{
        "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "url": str(url),
        "prediction": int(prediction),
        "result": "Phishing" if int(prediction) == 1 else "Legitimate",
        "phishing_probability": (
            round(float(probability), 6) if probability is not None else None
        )
    }])

    record.to_csv(
        HISTORY_PATH,
        mode="a",
        header=not os.path.exists(HISTORY_PATH) or os.path.getsize(HISTORY_PATH) == 0,
        index=False
    )


def load_scan_history():
    """Read saved scan history, returning an empty table if none exists."""
    columns = [
        "timestamp", "url", "prediction", "result", "phishing_probability"
    ]
    if not os.path.exists(HISTORY_PATH):
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(HISTORY_PATH)
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame(columns=columns)


def dashboard_domain_key(value):
    """Normalize a URL or hostname for matching optional security checks."""
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    if "://" not in raw_value:
        raw_value = "https://" + raw_value
    hostname = (urlsplit(raw_value).hostname or "").lower().strip(".")
    if not hostname:
        return None
    try:
        return get_registered_domain(hostname)
    except (NameError, ValueError):
        return hostname.removeprefix("www.")



def get_feature_importance_for_scan(model, features, top_n=6):
    """
    Return model-wide feature importance for the current scan's features.

    This is global model importance, not a per-URL causal explanation.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["Feature", "Value", "Model importance"])

    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
    else:
        names = list(features.keys())

    if len(names) != len(importances):
        return pd.DataFrame(columns=["Feature", "Value", "Model importance"])

    rows = []
    for name, importance in zip(names, importances):
        if name in features:
            rows.append({
                "Feature": str(name),
                "Value": str(features[name]),
                "Model importance": float(importance)
            })

    if not rows:
        return pd.DataFrame(columns=["Feature", "Value", "Model importance"])

    return (
        pd.DataFrame(rows)
        .sort_values("Model importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )



# --------------------------------------------------
# FEATURE 9: TYPOSQUATTING CHECK
# --------------------------------------------------

# Small, transparent list of well-known domains used for similarity screening.
# This is a heuristic list, not a live threat-intelligence feed.
KNOWN_BRANDS = {
    "google": "google.com",
    "amazon": "amazon.com",
    "microsoft": "microsoft.com",
    "paypal": "paypal.com",
    "apple": "apple.com",
    "facebook": "facebook.com",
    "instagram": "instagram.com",
    "whatsapp": "whatsapp.com",
    "netflix": "netflix.com",
    "wikipedia": "wikipedia.org",
    "linkedin": "linkedin.com",
    "yahoo": "yahoo.com",
    "outlook": "outlook.com",
    "github": "github.com",
    "dropbox": "dropbox.com",
}

# Common digit substitutions used in lookalike domains.
CONFUSABLE_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
})


def extract_domain_label(value):
    """Extract a host's likely registrable label without contacting the site."""
    raw_value = str(value).strip()
    if "://" not in raw_value:
        raw_value = "http://" + raw_value

    parsed = urlsplit(raw_value)
    hostname = (parsed.hostname or "").strip(".").lower()

    if not hostname:
        raise ValueError("Enter a valid domain or complete URL.")

    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        pass

    hostname = hostname.removeprefix("www.")
    labels = [part for part in hostname.split(".") if part]

    if not labels:
        raise ValueError("Could not extract a domain name.")

    multi_part_suffixes = {
        "co.uk", "org.uk", "ac.uk", "com.au", "net.au", "org.au",
        "co.in", "firm.in", "net.in", "org.in", "co.nz", "com.br",
        "com.sg", "com.mx", "co.jp"
    }

    if len(labels) >= 3 and ".".join(labels[-2:]) in multi_part_suffixes:
        registrable_label = labels[-3]
    elif len(labels) >= 2:
        registrable_label = labels[-2]
    else:
        registrable_label = labels[0]

    return hostname, registrable_label


def check_typosquatting(value):
    """
    Compare the extracted domain label against a small known-brand list.
    Returns a heuristic match; it does not establish maliciousness.
    """
    hostname, domain_label = extract_domain_label(value)
    normalized_label = domain_label.translate(CONFUSABLE_TRANSLATION)

    best_match = None
    best_ratio = 0.0
    best_brand = None

    for brand, official_domain in KNOWN_BRANDS.items():
        brand = brand.strip()
        ratio = difflib.SequenceMatcher(
            None, normalized_label, brand
        ).ratio()

        # Exact normalized matches catch simple substitutions such as g00gle.
        exact_confusable = normalized_label == brand and domain_label != brand

        # Similarity threshold limits noisy matches; avoid matching very short labels.
        if exact_confusable or (
            len(normalized_label) >= 4
            and len(brand) >= 4
            and ratio >= 0.78
            and normalized_label != brand
        ):
            if exact_confusable:
                ratio = 1.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = official_domain
                best_brand = brand

    return {
        "hostname": hostname,
        "domain_label": domain_label,
        "is_lookalike": best_match is not None,
        "possible_brand": best_brand,
        "suggested_domain": best_match,
        "similarity": best_ratio
    }



# --------------------------------------------------
# FEATURE 10: DOMAIN AGE LOOKUP (RDAP)
# --------------------------------------------------

MULTI_PART_PUBLIC_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk",
    "com.au", "net.au", "org.au",
    "co.in", "firm.in", "net.in", "org.in",
    "co.nz", "com.br", "com.sg", "com.mx", "co.jp"
}


def get_registered_domain(value):
    """Return a best-effort registrable domain for an input URL or hostname."""
    raw_value = str(value).strip()
    if "://" not in raw_value:
        raw_value = "https://" + raw_value

    parsed = urlsplit(raw_value)
    hostname = (parsed.hostname or "").strip(".").lower()
    if not hostname:
        raise ValueError("Enter a valid domain or complete URL.")

    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        pass

    hostname = hostname.removeprefix("www.")
    labels = [part for part in hostname.split(".") if part]

    if len(labels) < 2:
        raise ValueError("Enter a public domain name, such as example.com.")

    last_two = ".".join(labels[-2:])
    if len(labels) >= 3 and last_two in MULTI_PART_PUBLIC_SUFFIXES:
        return ".".join(labels[-3:])

    return ".".join(labels[-2:])


def lookup_domain_registration(value):
    """
    Query the public RDAP service for registration events.
    The domain is sent to rdap.org and the relevant registry RDAP server.
    """
    domain = get_registered_domain(value)
    endpoint = f"https://rdap.org/domain/{domain}"
    request = urllib.request.Request(
        endpoint,
        headers={
            "Accept": "application/rdap+json, application/json",
            "User-Agent": "SURF-Safe-Educational-DomainAgeChecker/1.0"
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                "No RDAP registration record was found. The domain may be "
                "unsupported by RDAP, unregistered, or unavailable from the registry."
            ) from exc
        raise ValueError(f"RDAP lookup returned HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError(
            "Could not reach the RDAP service. Check your internet connection "
            "and try again."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("The RDAP service returned an unreadable response.") from exc

    event_dates = {}
    for event in payload.get("events", []):
        action = str(event.get("eventAction", "")).lower()
        date_text = event.get("eventDate")
        if not date_text:
            continue
        try:
            parsed_date = datetime.fromisoformat(
                str(date_text).replace("Z", "+00:00")
            )
            if parsed_date.tzinfo is None:
                parsed_date = parsed_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
            event_dates[action] = parsed_date
        except ValueError:
            continue

    registration_date = (
        event_dates.get("registration")
        or event_dates.get("registered")
        or event_dates.get("creation")
    )
    expiration_date = (
        event_dates.get("expiration")
        or event_dates.get("expiry")
    )

    if registration_date is None:
        raise ValueError(
            "The RDAP record did not provide a registration date, so the "
            "domain's age cannot be calculated."
        )

    now = datetime.now().astimezone()
    age_days = max(0, (now - registration_date.astimezone(now.tzinfo)).days)

    return {
        "domain": domain,
        "registration_date": registration_date,
        "expiration_date": expiration_date,
        "age_days": age_days,
        "age_years": age_days / 365.25,
        "registrar": payload.get("entities", [{}])[0].get("vcardArray", [None, []])[0]
        if payload.get("entities") else None,
        "rdap_status": payload.get("status", [])
    }



# --------------------------------------------------
# FEATURE 11: SAFE REDIRECT CHAIN ANALYSIS
# --------------------------------------------------

def validate_public_http_url(value):
    """Allow only public HTTP(S) destinations on standard ports."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Enter a URL to inspect.")
    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded usernames or passwords are not allowed.")

    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise ValueError("The URL does not contain a valid hostname.")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("Local and internal hostnames are blocked.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc

    allowed_port = 443 if parsed.scheme.lower() == "https" else 80
    if port not in {None, allowed_port}:
        raise ValueError("Only standard HTTP (80) and HTTPS (443) ports are allowed.")

    # Block literal IP addresses and resolve hostnames before connecting.
    try:
        ipaddress.ip_address(hostname)
        raise ValueError("Direct IP-address URLs are blocked.")
    except ValueError as exc:
        if "Direct IP-address" in str(exc):
            raise

    try:
        addresses = socket.getaddrinfo(
            hostname, allowed_port, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError("The hostname could not be resolved.") from exc

    if not addresses:
        raise ValueError("The hostname did not resolve to an IP address.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError(
                "The URL resolves to a private, local, or reserved IP address and was blocked."
            )

    return raw


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect destination and record the chain."""

    def __init__(self):
        super().__init__()
        self.redirect_chain = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        safe_url = validate_public_http_url(newurl)
        self.redirect_chain.append({
            "status": int(code),
            "from": req.full_url,
            "to": safe_url
        })
        if len(self.redirect_chain) > 8:
            raise urllib.error.HTTPError(
                req.full_url, code, "Redirect limit exceeded", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, safe_url)


def inspect_redirect_chain(value):
    """Fetch headers/body minimally and follow a limited, validated redirect chain."""
    start_url = validate_public_http_url(value)
    redirect_handler = SafeRedirectHandler()
    opener = urllib.request.build_opener(redirect_handler)

    request = urllib.request.Request(
        start_url,
        headers={
            "User-Agent": "SURF-Safe-Redirect-Checker/1.0",
            "Range": "bytes=0-0"
        },
        method="GET"
    )

    try:
        with opener.open(request, timeout=10) as response:
            status_code = int(response.status)
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "Not supplied")
    except urllib.error.HTTPError as exc:
        # HTTP error statuses still provide useful redirect/final destination data.
        status_code = int(exc.code)
        final_url = exc.geturl() or start_url
        content_type = exc.headers.get("Content-Type", "Not supplied") if exc.headers else "Not supplied"
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ValueError(f"Could not inspect this URL: {reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ValueError("The website did not respond before the 10-second timeout.") from exc

    # Validate final URL again before displaying it.
    final_url = validate_public_http_url(final_url)

    return {
        "start_url": start_url,
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "redirects": redirect_handler.redirect_chain,
        "redirect_count": len(redirect_handler.redirect_chain)
    }



# --------------------------------------------------
# FEATURE 12: SSL CERTIFICATE CHECKER
# --------------------------------------------------

def inspect_ssl_certificate(value):
    """Inspect a public hostname's TLS certificate without fetching page content."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Enter a domain or HTTPS URL.")

    if "://" not in raw:
        raw = "https://" + raw

    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Enter an HTTP or HTTPS URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded usernames or passwords are not allowed.")

    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise ValueError("The URL does not contain a valid hostname.")
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("Local and internal hostnames are blocked.")

    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("The hostname could not be processed.") from exc

    try:
        ipaddress.ip_address(hostname)
        raise ValueError("Direct IP-address checks are not supported. Enter a domain name.")
    except ValueError as exc:
        if "Direct IP-address" in str(exc):
            raise

    # Resolve first and reject any non-public answer. Connect to the validated
    # address directly to reduce the risk of a DNS change between validation and connection.
    try:
        address_records = socket.getaddrinfo(
            hostname, 443, type=socket.SOCK_STREAM
        )
    except socket.gaierror as exc:
        raise ValueError("The hostname could not be resolved.") from exc

    if not address_records:
        raise ValueError("The hostname did not resolve to an IP address.")

    public_records = []
    for record in address_records:
        ip_text = record[4][0]
        ip_obj = ipaddress.ip_address(ip_text)
        if not ip_obj.is_global:
            raise ValueError(
                "The hostname resolves to a private, local, or reserved IP address and was blocked."
            )
        public_records.append(record)

    tls_context = ssl.create_default_context()
    last_error = None
    tls_socket = None

    for family, socktype, proto, canonname, sockaddr in public_records:
        raw_socket = socket.socket(family, socktype, proto)
        raw_socket.settimeout(8)
        try:
            raw_socket.connect(sockaddr)
            tls_socket = tls_context.wrap_socket(
                raw_socket, server_hostname=hostname
            )
            break
        except (OSError, ssl.SSLError) as exc:
            last_error = exc
            raw_socket.close()

    if tls_socket is None:
        raise ValueError(f"Could not establish a trusted TLS connection: {last_error}")

    try:
        with tls_socket:
            certificate = tls_socket.getpeercert()
            cipher_info = tls_socket.cipher()
            tls_version = tls_socket.version()

        if not certificate:
            raise ValueError("The server did not provide a readable certificate.")

        not_before_text = certificate.get("notBefore")
        not_after_text = certificate.get("notAfter")
        if not_after_text is None:
            raise ValueError("The certificate did not include an expiration date.")

        not_before = (
            datetime.strptime(not_before_text, "%b %d %H:%M:%S %Y %Z")
            if not_before_text else None
        )
        not_after = datetime.strptime(
            not_after_text, "%b %d %H:%M:%S %Y %Z"
        )

        now_utc = datetime.utcnow()
        days_remaining = (not_after - now_utc).days
        is_currently_valid = (
            (not_before is None or now_utc >= not_before)
            and now_utc < not_after
        )

        def flatten_name(name_items):
            parts = []
            for group in name_items or ():
                for key, val in group:
                    if key in {"commonName", "organizationName"}:
                        parts.append(str(val))
            return ", ".join(dict.fromkeys(parts)) or "Not provided"

        subject = flatten_name(certificate.get("subject", ()))
        issuer = flatten_name(certificate.get("issuer", ()))
        san_entries = [
            value for kind, value in certificate.get("subjectAltName", ())
            if kind == "DNS"
        ]

        return {
            "hostname": hostname,
            "subject": subject,
            "issuer": issuer,
            "not_before": not_before,
            "not_after": not_after,
            "days_remaining": days_remaining,
            "is_valid": is_currently_valid,
            "tls_version": tls_version or "Not reported",
            "cipher": cipher_info[0] if cipher_info else "Not reported",
            "san": san_entries[:12]
        }
    except (ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("The certificate"):
            raise
        raise ValueError(f"Could not read the certificate details: {exc}") from exc



# --------------------------------------------------
# FEATURE 13: DNS SECURITY CHECKER
# --------------------------------------------------

def normalize_dns_domain(value):
    """Convert a URL or hostname to a validated ASCII domain name."""
    raw_value = str(value).strip()
    if not raw_value:
        raise ValueError("Enter a domain or URL.")

    if "://" not in raw_value:
        raw_value = "https://" + raw_value

    parsed = urlsplit(raw_value)
    hostname = (parsed.hostname or "").strip(".").lower()

    if not hostname:
        raise ValueError("Could not extract a valid domain name.")

    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("The domain name contains invalid characters.") from exc

    if len(hostname) > 253 or "." not in hostname:
        raise ValueError("Enter a public domain name, such as example.com.")

    labels = hostname.split(".")
    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not all(ch.isalnum() or ch == "-" for ch in label)
        for label in labels
    ):
        raise ValueError("The domain name format is invalid.")

    return hostname


def query_dns_over_https(name, record_type):
    """
    Query public DNS-over-HTTPS providers in order:
    Cloudflare first, then Google DNS as a fallback.
    The queried domain is sent to the provider; no website page is opened.
    """
    query = urlencode({"name": name, "type": record_type})
    providers = [
        (
            "Cloudflare",
            f"https://cloudflare-dns.com/dns-query?{query}",
            {"Accept": "application/dns-json"}
        ),
        (
            "Google Public DNS",
            f"https://dns.google/resolve?{query}",
            {"Accept": "application/dns-json"}
        ),
    ]

    errors = []
    for provider_name, endpoint, extra_headers in providers:
        headers = {
            "User-Agent": "SURF-Safe-DNS-Security-Checker/1.1",
            **extra_headers
        }
        request = urllib.request.Request(endpoint, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))

            status = payload.get("Status")
            if status not in (0, 3):
                errors.append(f"{provider_name}: DNS response code {status}")
                continue

            answers = []
            for answer in payload.get("Answer", []):
                data = str(answer.get("data", "")).strip()
                if data:
                    answers.append(data)

            # NXDOMAIN/no-answer is a valid DNS result, not a connection failure.
            return answers

        except urllib.error.HTTPError as exc:
            errors.append(f"{provider_name}: HTTP {exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{provider_name}: connection failed")
        except json.JSONDecodeError:
            errors.append(f"{provider_name}: unreadable response")

    raise ValueError(
        "Could not reach either public DNS service. "
        "Cloudflare and Google DNS both failed. Check your internet connection, "
        "VPN, firewall, or antivirus and try again. Details: "
        + "; ".join(errors)
    )


def inspect_dns_security(value):
    """Collect common DNS records and evaluate basic SPF/DMARC presence."""
    domain = normalize_dns_domain(value)

    record_types = {
        "A": (domain, "A"),
        "AAAA": (domain, "AAAA"),
        "MX": (domain, "MX"),
        "NS": (domain, "NS"),
        "TXT": (domain, "TXT"),
        "DMARC": (f"_dmarc.{domain}", "TXT")
    }

    records = {}
    for key, (query_name, record_type) in record_types.items():
        records[key] = query_dns_over_https(query_name, record_type)

    txt_records = records["TXT"]
    spf_records = [
        record for record in txt_records
        if record.strip('"').lower().startswith("v=spf1")
    ]
    dmarc_records = [
        record for record in records["DMARC"]
        if "v=dmarc1" in record.lower()
    ]

    return {
        "domain": domain,
        "records": records,
        "spf_records": spf_records,
        "dmarc_records": dmarc_records,
        "has_a": bool(records["A"]),
        "has_aaaa": bool(records["AAAA"]),
        "has_mx": bool(records["MX"]),
        "has_ns": bool(records["NS"]),
        "has_spf": bool(spf_records),
        "has_dmarc": bool(dmarc_records)
    }


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SURF-Safe | URL Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d18;
    --panel: #101827;
    --panel-light: #151f31;
    --border: #253249;
    --text: #edf4ff;
    --muted: #8d9bb2;
    --green: #39e6a5;
    --blue: #5ca9ff;
    --red: #ff647c;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #14233a 0%, transparent 40%),
        radial-gradient(ellipse at 90% 20%, #102b32 0%, transparent 30%),
        var(--bg);
    color: var(--text);
}

/* Keep the sidebar controls visible and blend Streamlit's header into
   the dark dashboard instead of showing a white strip. */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .stAppHeader,
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: #080d18 !important;
    background-color: #080d18 !important;
    border: 0 !important;
    box-shadow: none !important;
}

.stAppHeader,
header[data-testid="stHeader"] {
    display: block !important;
    height: 2.5rem !important;
}

[data-testid="stToolbar"] {
    display: flex !important;
}

/* Keep the sidebar itself visible and styled. */
[data-testid="stSidebar"] {
    background: #0b1220 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

.block-container {
    max-width: 1450px;
    padding-top: 0.8rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
}

p, label, li {
    color: var(--muted);
}

/* Sidebar branding */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 25px 0;
}

.brand-icon {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    background: linear-gradient(135deg, #183d49, #142c46);
    border: 1px solid #2e6970;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}

.brand-name {
    color: #f2f7ff;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 21px;
    letter-spacing: -0.7px;
}

.brand-sub {
    font-size: 10px;
    color: var(--green);
    letter-spacing: 2px;
    margin-top: 3px;
}

.side-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #63738c;
    margin: 25px 0 12px 0;
}

.side-status {
    background: #111d2d;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-top: 15px;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 12px rgba(57, 230, 165, 0.55);
}

/* Hero */

.hero {
    padding: 10px 0 24px 0;
}

.eyebrow {
    color: var(--green);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    margin-bottom: 14px;
}

.hero-title {
    color: #f4f7ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(34px, 4vw, 49px);
    font-weight: 700;
    line-height: 1.12;
    letter-spacing: -2px;
    margin: 0;
}

.hero-title span {
    background: linear-gradient(90deg, #39e6a5, #5ca9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: #8d9bb2;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 15px;
    max-width: 650px;
}

/* Panels */

.panel {
    background: linear-gradient(145deg, #111c2d, #0d1523);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
}

.panel-heading {
    color: #edf4ff;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 5px;
}

.panel-sub {
    color: #8190a8;
    font-size: 12px;
    line-height: 1.6;
}

.section-kicker {
    color: var(--blue);
    font-size: 10px;
    letter-spacing: 2px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Metric cards */

.metric-card {
    background: linear-gradient(145deg, #131f31, #0e1726);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 19px;
    min-height: 112px;
}

.metric-label {
    color: #8d9bb2;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-value {
    color: #f2f7ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 27px;
    font-weight: 700;
    margin-top: 12px;
}

.metric-note {
    color: #73849d;
    font-size: 11px;
    margin-top: 4px;
}

/* Result cards */

.result-card {
    border-radius: 16px;
    padding: 23px;
    margin: 10px 0 18px 0;
}

.result-safe {
    background: linear-gradient(120deg, #0c2b28, #10241f);
    border: 1px solid #216b55;
}

.result-danger {
    background: linear-gradient(120deg, #351822, #241621);
    border: 1px solid #8e3549;
}

.result-title {
    font-size: 22px;
    font-weight: 800;
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 7px;
}

.result-description {
    color: #b1bdcf;
    font-size: 12px;
    line-height: 1.7;
    overflow-wrap: anywhere;
}

.risk-pill {
    display: inline-block;
    border-radius: 30px;
    padding: 6px 12px;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 800;
    margin-bottom: 13px;
}

.risk-high {
    background: #4b1d2a;
    color: #ff8093;
    border: 1px solid #8e3549;
}

.risk-low {
    background: #12382e;
    color: #5aefb8;
    border: 1px solid #216b55;
}

.feature-chip {
    display: inline-block;
    padding: 7px 10px;
    background: #17253a;
    border: 1px solid #293b55;
    color: #a9caff;
    border-radius: 8px;
    font-size: 11px;
    margin: 4px 4px 4px 0;
}

/* Inputs */

.stTextInput > div > div > input {
    background: #0b1422 !important;
    color: #eff6ff !important;
    border: 1px solid #30415b !important;
    border-radius: 10px !important;
    min-height: 52px;
}

.stTextInput > div > div > input:focus {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 1px var(--green) !important;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    border: 1px solid #49e9b0;
    border-radius: 10px;
    background: linear-gradient(100deg, #35dca0, #54c9d8);
    color: #07141a;
    font-size: 14px;
    font-weight: 800;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: #86ffd5;
    color: #07141a;
    box-shadow: 0 0 22px rgba(57, 230, 165, 0.16);
    transform: translateY(-1px);
}

div[data-testid="stExpander"] {
    background: #0f1929;
    border: 1px solid var(--border);
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

hr {
    border-color: var(--border);
}

.footer {
    text-align: center;
    color: #61718a;
    font-size: 11px;
    padding: 22px 0 5px 0;
    letter-spacing: 0.3px;
}

div[data-testid="stMetric"] {
    background: #111c2d;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #8d9bb2;
}

div[data-testid="stMetricValue"] {
    color: #edf4ff;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #39e6a5, #5ca9ff);
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div>
            <div class="brand-name">SURF-Safe</div>
            <div class="brand-sub">URL THREAT INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">WORKSPACE</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background: linear-gradient(90deg, #17352f, #11251f);
        border: 1px solid #285947;
        padding: 13px;
        border-radius: 10px;
        color: #63edb7;
        font-size: 13px;
        font-weight: 600;
    ">◈ &nbsp; URL Security Scanner</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">SYSTEM INFORMATION</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="side-status">
        <div style="color:#e9f1ff;font-size:12px;font-weight:600;">
            <span class="status-dot"></span>Scanner Interface
        </div>
        <div style="color:#7f91a9;font-size:11px;margin:9px 0 0 16px;">
            Ready for analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(MODEL_PATH):
        st.markdown("""
        <div class="side-status">
            <div style="color:#e9f1ff;font-size:12px;font-weight:600;">
                <span class="status-dot"></span>ML Model File
            </div>
            <div style="color:#7f91a9;font-size:11px;margin:9px 0 0 16px;">
                Model file detected
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="side-status">
            <div style="color:#ff8093;font-size:12px;font-weight:600;">
                Model file not found
            </div>
            <div style="color:#7f91a9;font-size:11px;margin-top:8px;">
                Train the model before scanning.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="side-label">ABOUT THIS PROJECT</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#8999b1;font-size:12px;line-height:1.9;">
        An explainable machine-learning application for
        URL-based phishing risk assessment.
        <br><br>
        <span style="color:#4ce5ae;">●</span>
        Feature extraction
        <br>
        <span style="color:#4ce5ae;">●</span>
        ML-based classification
        <br>
        <span style="color:#4ce5ae;">●</span>
        Prediction explanations
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.caption("SURF-Safe  ·  ACADEMIC PROJECT")
    st.caption("Machine Learning • Cybersecurity")


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">◈ &nbsp; CYBER DEFENSE / URL ANALYTICS</div>
    <div class="hero-title">
        Detect threats.<br>
        <span>Protect your digital world.</span>
    </div>
    <div class="hero-desc">
        Analyze suspicious URLs with machine learning.
        Identify potential phishing indicators and understand
        the features behind each prediction.
    </div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# TOP METRIC CARDS
# --------------------------------------------------

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Detection Engine</div>
        <div class="metric-value" style="font-size:23px;">
            ML Powered
        </div>
        <div class="metric-note">Trained classification model</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Analysis Mode</div>
        <div class="metric-value" style="font-size:23px;">
            URL Scan
        </div>
        <div class="metric-note">Feature-based inspection</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Explainability</div>
        <div class="metric-value" style="font-size:23px;">
            Enabled
        </div>
        <div class="metric-note">Human-readable indicators</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

if not os.path.exists(MODEL_PATH):

    st.error(
        "The trained model was not found. "
        "Run `python src/train_model.py` and try again."
    )
    st.stop()


# --------------------------------------------------
# URL SCANNER
# --------------------------------------------------

left, right = st.columns([1.45, 1], gap="large")

with left:

    st.markdown("""
    <div class="panel">
        <div class="section-kicker">01 / INPUT</div>
        <div class="panel-heading">URL Threat Scanner</div>
        <div class="panel-sub">
            Enter the complete URL you want to inspect.
            The model will analyze its extracted features.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature 7: QR-code URL extraction. The extracted URL is placed
    # into the regular scanner; the user must still press ANALYZE URL.
    with st.expander("▦  Scan a QR code image", expanded=False):
        st.caption(
            "Upload a QR-code image. SURF-Safe extracts the text locally "
            "and places a web URL into the scanner. It does not open the URL."
        )
        qr_image = st.file_uploader(
            "Choose a QR code image",
            type=["png", "jpg", "jpeg", "webp"],
            key="qr_code_upload"
        )
        decode_qr_clicked = st.button(
            "Extract URL from QR code",
            key="decode_qr_button",
            use_container_width=True
        )

        if decode_qr_clicked:
            if qr_image is None:
                st.warning("Please upload a QR-code image first.")
            else:
                try:
                    import cv2
                    import numpy as np

                    image_bytes = np.frombuffer(qr_image.getvalue(), dtype=np.uint8)
                    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

                    if image is None:
                        st.error("The uploaded image could not be read.")
                    else:
                        detector = cv2.QRCodeDetector()
                        decoded_text, points, _ = detector.detectAndDecode(image)
                        decoded_text = decoded_text.strip()

                        if not decoded_text:
                            st.error(
                                "No readable QR code was found. Try a clearer "
                                "image with the full QR code visible."
                            )
                        else:
                            from urllib.parse import urlsplit

                            parsed = urlsplit(decoded_text)
                            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                                st.warning(
                                    "The QR code contains text, but it is not a "
                                    "complete HTTP/HTTPS URL. It was not added "
                                    "to the URL scanner."
                                )
                            else:
                                st.session_state["target_url_input"] = decoded_text
                                st.success(
                                    "URL extracted. Review it in the URL field "
                                    "below, then click ANALYZE URL."
                                )
                                st.rerun()

                except ImportError:
                    st.error(
                        "QR scanning needs OpenCV and NumPy. In your VS Code "
                        "terminal run: pip install opencv-python-headless numpy"
                    )
                except Exception as exc:
                    st.error(f"Could not decode this QR image: {exc}")

    with st.form("url_scanner_form"):

        url = st.text_input(
            "TARGET URL",
            placeholder="https://example.com/login",
            help="Enter a URL for local model-based analysis.",
            key="target_url_input"
        )

        submitted = st.form_submit_button(
            "⚡  ANALYZE URL"
        )

    st.markdown("""
    <div style="
        padding:12px 15px;
        background:#101a29;
        border:1px solid #253249;
        border-radius:10px;
        color:#8d9bb2;
        font-size:11px;
        line-height:1.8;
    ">
        <span style="color:#56dcb1;font-weight:700;">PRIVACY NOTE</span>
        <br>
        This interface performs feature extraction and prediction
        using your local Python model. No external threat-intelligence
        lookup is implemented in this application.
    </div>
    """, unsafe_allow_html=True)


with right:

    st.markdown("""
    <div class="panel">
        <div class="section-kicker">02 / HOW IT WORKS</div>
        <div class="panel-heading">Analysis Pipeline</div>
        <div class="panel-sub">
            Three stages from URL input to an interpretable result.
        </div>
        <br>
        <div style="
            padding:13px;
            border:1px solid #29384f;
            background:#111e30;
            border-radius:10px;
            margin-bottom:10px;
        ">
            <div style="color:#56dcb1;font-size:10px;font-weight:800;">
                STAGE 01
            </div>
            <div style="color:#edf4ff;font-size:13px;font-weight:700;
                        margin-top:5px;">
                ◈ &nbsp; Feature Extraction
            </div>
            <div style="color:#8798b0;font-size:11px;margin-top:5px;">
                Inspect URL characteristics.
            </div>
        </div>
        <div style="
            padding:13px;
            border:1px solid #29384f;
            background:#111e30;
            border-radius:10px;
            margin-bottom:10px;
        ">
            <div style="color:#68b5ff;font-size:10px;font-weight:800;">
                STAGE 02
            </div>
            <div style="color:#edf4ff;font-size:13px;font-weight:700;
                        margin-top:5px;">
                ⌘ &nbsp; ML Classification
            </div>
            <div style="color:#8798b0;font-size:11px;margin-top:5px;">
                Generate a model prediction.
            </div>
        </div>
        <div style="
            padding:13px;
            border:1px solid #29384f;
            background:#111e30;
            border-radius:10px;
        ">
            <div style="color:#c29aff;font-size:10px;font-weight:800;">
                STAGE 03
            </div>
            <div style="color:#edf4ff;font-size:13px;font-weight:700;
                        margin-top:5px;">
                ◎ &nbsp; Explainable Results
            </div>
            <div style="color:#8798b0;font-size:11px;margin-top:5px;">
                Display the decision and URL indicators.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    if not url.strip():
        st.warning("Please enter a URL before starting the scan.")

    else:
        try:
            with st.spinner("Analyzing URL features and evaluating model..."):

                model = joblib.load(MODEL_PATH)

                features = extract_features(url.strip())
                X = pd.DataFrame([features])

                # Align feature columns if the model exposes
                # its original training feature names.
                if hasattr(model, "feature_names_in_"):
                    X = X.reindex(
                        columns=model.feature_names_in_,
                        fill_value=0
                    )

                prediction = int(model.predict(X)[0])

                probability = None

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(X)[0]
                    classes = list(model.classes_)

                    if 1 in classes:
                        probability = float(
                            probabilities[classes.index(1)]
                        )

                raw_reasons = explain(features)
                if raw_reasons is None:
                    reasons = []
                elif isinstance(raw_reasons, str):
                    reasons = [raw_reasons]
                else:
                    reasons = list(raw_reasons)

                # The explain() helper is a separate heuristic check. Remove
                # generic "no lexical warning" text when the ML model predicts
                # phishing, because that wording can look contradictory.
                if prediction == 1:
                    reasons = [
                        reason for reason in reasons
                        if "no strong lexical warning signs" not in str(reason).lower()
                        and "no suspicious lexical" not in str(reason).lower()
                        and "no lexical warning" not in str(reason).lower()
                    ]

                # These are model-wide importance values, not per-URL causal effects.
                importance_df = get_feature_importance_for_scan(
                    model, features, top_n=6
                )

                # Feature 4: persist each successful scan locally.
                save_scan_history(url.strip(), prediction, probability)

                # Keep the latest model result available to the combined
                # Security Risk Score dashboard below.
                st.session_state["sentinel_last_ml_scan"] = {
                    "url": url.strip(),
                    "prediction": prediction,
                    "probability": probability,
                    "timestamp": datetime.now().astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S %Z"
                    )
                }

            st.markdown("---")

            st.markdown("""
            <div class="eyebrow">◈ &nbsp; ANALYSIS COMPLETE / REPORT</div>
            """, unsafe_allow_html=True)

            result_left, result_right = st.columns(
                [1.3, 0.7],
                gap="large"
            )

            # --------------------------
            # RESULT CARD
            # --------------------------

            with result_left:

                safe_url = html.escape(url.strip())

                if prediction == 1:

                    st.markdown(f"""
                    <div class="result-card result-danger">
                        <div class="risk-pill risk-high">
                            ⚠ &nbsp; POTENTIAL THREAT DETECTED
                        </div>
                        <div class="result-title" style="color:#ff8093;">
                            Phishing Suspected
                        </div>
                        <div class="result-description">
                            The machine-learning model classified
                            this URL as potentially malicious.
                            Avoid entering sensitive information until
                            the URL has been independently verified.
                            <br><br>
                            <span style="color:#ff8093;">
                                TARGET:
                            </span>
                            {safe_url}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown(f"""
                    <div class="result-card result-safe">
                        <div class="risk-pill risk-low">
                            ✓ &nbsp; NO PHISHING CLASSIFICATION
                        </div>
                        <div class="result-title" style="color:#5aefb8;">
                            No Phishing Detected
                        </div>
                        <div class="result-description">
                            The machine-learning model classified
                            this URL as legitimate. This does not
                            guarantee that the website is safe.
                            <br><br>
                            <span style="color:#5aefb8;">
                                TARGET:
                            </span>
                            {safe_url}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # --------------------------
            # PROBABILITY CARD
            # --------------------------

            with result_right:

                if probability is not None:

                    st.markdown("""
                    <div class="panel">
                        <div class="section-kicker">MODEL OUTPUT</div>
                        <div class="panel-heading">
                            Phishing Class Score
                        </div>
                        <div class="panel-sub">
                            Probability reported for class 1.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div style="
                            font-family:'Space Grotesk',sans-serif;
                            font-size:43px;
                            font-weight:700;
                            color:{'#ff8093' if prediction == 1 else '#5aefb8'};
                            margin:0 0 5px 0;
                        ">{probability:.1%}</div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(probability)

                    st.caption(
                        "Model probability for the phishing class. "
                        "This is not a guarantee of actual risk."
                    )

                    # Feature 5: convert the model's phishing probability
                    # into an easy-to-read 0-100 score. This is a model
                    # estimate, not a verified real-world threat rating.
                    risk_score = int(round(probability * 100))

                    if risk_score < 30:
                        risk_level = "LOW"
                        risk_color = "#5aefb8"
                    elif risk_score < 60:
                        risk_level = "MODERATE"
                        risk_color = "#ffd166"
                    elif risk_score < 85:
                        risk_level = "HIGH"
                        risk_color = "#ff9f5a"
                    else:
                        risk_level = "CRITICAL"
                        risk_color = "#ff8093"

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div class="section-kicker">FEATURE 5 / RISK ASSESSMENT</div>
                            <div class="panel-heading">Phishing Risk Score</div>
                            <div style="display:flex;align-items:baseline;gap:8px;margin-top:12px;">
                                <span style="font-family:'Space Grotesk',sans-serif;
                                             font-size:42px;font-weight:700;
                                             color:{risk_color};">{risk_score}</span>
                                <span style="color:#8d9bb2;font-size:14px;">/ 100</span>
                            </div>
                            <div style="display:inline-block;margin-top:5px;
                                        padding:6px 11px;border-radius:20px;
                                        color:{risk_color};border:1px solid {risk_color};
                                        font-size:11px;font-weight:800;letter-spacing:1px;">
                                {risk_level} MODEL SCORE
                            </div>
                            <div style="color:#8d9bb2;font-size:11px;
                                        line-height:1.7;margin-top:12px;">
                                Based on the model's estimated phishing probability.
                                It is not a verified threat-intelligence rating
                                or a guarantee that a website is safe or malicious.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(risk_score / 100)

                else:
                    risk_score = None
                    st.info(
                        "A 0-100 risk score cannot be calculated because "
                        "this model does not provide prediction probabilities."
                    )


                    st.info(
                        "This model does not support predict_proba(). "
                        "The classification is available, but a "
                        "probability score cannot be displayed."
                    )

            # --------------------------
            # SUMMARY METRICS
            # --------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="section-kicker">
                03 / SCAN SUMMARY
            </div>
            """, unsafe_allow_html=True)

            s1, s2, s3 = st.columns(3)

            with s1:
                st.metric(
                    "Extracted Features",
                    len(features)
                )

            with s2:
                st.metric(
                    "Classification",
                    "Phishing" if prediction == 1 else "Legitimate"
                )

            with s3:
                st.metric(
                    "Explanation Items",
                    len(reasons)
                )

            # --------------------------
            # EXPLANATION
            # --------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class="panel">
                <div class="section-kicker">04 / EXPLAINABILITY</div>
                <div class="panel-heading">
                    Why did the model make this prediction?
                </div>
                <div class="panel-sub">
                    Heuristic URL indicators and model-wide feature importance.
                    The ML prediction is the classification result; heuristic
                    text checks provide supporting context only.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if reasons:
                st.markdown(
                    "**Indicators returned by the explanation function**"
                )

                for i, reason in enumerate(reasons, start=1):
                    st.markdown(
                        f"""
                        <div style="
                            display:flex;
                            align-items:flex-start;
                            gap:13px;
                            padding:15px;
                            margin-bottom:9px;
                            border:1px solid #253249;
                            background:#101a29;
                            border-radius:11px;
                        ">
                            <div style="
                                min-width:30px;
                                height:30px;
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                border-radius:9px;
                                background:#19352f;
                                color:#5aefb8;
                                font-size:12px;
                                font-weight:800;
                            ">{i:02d}</div>
                            <div style="
                                color:#d7e2f3;
                                font-size:13px;
                                line-height:1.8;
                                overflow-wrap:anywhere;
                            ">{html.escape(str(reason))}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                if prediction == 1:
                    st.warning(
                        "The machine-learning model classified this URL as "
                        "phishing. The separate lexical explanation helper did "
                        "not identify a specific keyword-based warning for this "
                        "URL. This does not cancel or weaken the model's result; "
                        "the prediction uses the combined extracted feature values."
                    )
                else:
                    st.info(
                        "The explanation helper did not return a specific "
                        "text indicator for this URL. See the extracted feature "
                        "values and model-wide feature importance below."
                    )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Most influential model features**")

            if not importance_df.empty:
                st.caption(
                    "These are the model's globally important features, "
                    "shown with this URL's extracted values. They describe "
                    "what the trained model relies on overall; they do not "
                    "prove that an individual feature caused this prediction."
                )

                importance_chart = importance_df.set_index("Feature")[
                    "Model importance"
                ]
                st.bar_chart(importance_chart, use_container_width=True)

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info(
                    "This saved model does not expose feature_importances_. "
                    "The text indicators above and extracted feature table "
                    "are still available."
                )

            # --------------------------
            # TECHNICAL DETAILS
            # --------------------------

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander(
                "⌘  View extracted URL features and technical details"
            ):

                st.markdown(
                    "These values were passed to the prediction model."
                )

                feature_df = pd.DataFrame({
                    "Feature": list(features.keys()),
                    "Value": [
                        str(value) for value in features.values()
                    ]
                })

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    label="⬇ Download Feature Report (CSV)",
                    data=feature_df.to_csv(index=False).encode("utf-8"),
                    file_name="url_feature_report.csv",
                    mime="text/csv"
                )

            # --------------------------
            # SAFETY DISCLAIMER
            # --------------------------

            st.markdown("""
            <div style="
                background:#151a27;
                border:1px solid #40374a;
                border-radius:11px;
                padding:16px;
                margin-top:20px;
            ">
                <div style="
                    color:#d9c1ff;
                    font-size:12px;
                    font-weight:700;
                ">⚠ &nbsp; SECURITY DISCLAIMER</div>
                <div style="
                    color:#a0aabe;
                    font-size:11px;
                    line-height:1.9;
                    margin-top:8px;
                ">
                    This application provides an ML-based prediction,
                    not a definitive security verdict. False positives
                    and false negatives are possible. A legitimate
                    classification does not guarantee website safety.
                    Avoid opening suspicious links simply to test them.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:

            st.error(
                "The scan could not be completed. "
                "Check your feature extraction function, model "
                "feature columns, and prediction explanation code."
            )

            with st.expander("Technical error details"):
                st.code(str(e))



# --------------------------------------------------
# FEATURE 8: BULK URL SCANNER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 8 / BATCH ANALYSIS</div>
<div class="panel-heading">Bulk URL Scanner</div>
<div class="panel-sub">
    Upload a CSV file containing URLs. SURF-Safe analyses each URL locally
    with the trained model. URLs are not opened or visited.
</div>
""", unsafe_allow_html=True)

bulk_file = st.file_uploader(
    "Upload a CSV file containing URLs",
    type=["csv"],
    key="bulk_url_csv_upload"
)

if bulk_file is not None:
    try:
        bulk_input_df = pd.read_csv(bulk_file)

        if bulk_input_df.empty:
            st.warning("The uploaded CSV has no data rows.")
        else:
            # Find a likely URL column, allowing common header variations.
            normalized_columns = {
                str(column).strip().lower(): column
                for column in bulk_input_df.columns
            }
            likely_names = ["url", "urls", "website", "link", "target_url"]
            detected_column = next(
                (normalized_columns[name] for name in likely_names
                 if name in normalized_columns),
                None
            )

            if detected_column is not None:
                default_index = list(bulk_input_df.columns).index(detected_column)
            else:
                default_index = 0

            selected_url_column = st.selectbox(
                "Select the column containing URLs",
                options=list(bulk_input_df.columns),
                index=default_index,
                key="bulk_url_column"
            )

            st.caption(
                f"{len(bulk_input_df):,} rows found. "
                "For this first version, up to 500 non-empty URLs can be "
                "processed in one batch."
            )

            if st.button(
                "Run Bulk URL Analysis",
                key="run_bulk_url_analysis",
                use_container_width=True
            ):
                raw_urls = (
                    bulk_input_df[selected_url_column]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                raw_urls = raw_urls[raw_urls != ""].drop_duplicates().tolist()

                if not raw_urls:
                    st.warning("No non-empty URLs were found in that column.")
                else:
                    if len(raw_urls) > 500:
                        st.warning(
                            "Only the first 500 unique non-empty URLs will be "
                            "processed in this batch."
                        )
                        raw_urls = raw_urls[:500]

                    bulk_model = joblib.load(MODEL_PATH)
                    bulk_rows = []
                    progress = st.progress(0, text="Preparing bulk analysis...")

                    from urllib.parse import urlsplit

                    for index, candidate_url in enumerate(raw_urls, start=1):
                        row = {
                            "url": candidate_url,
                            "result": "Error",
                            "prediction": None,
                            "phishing_probability": None,
                            "risk_score": None,
                            "status": "Not analysed"
                        }

                        try:
                            parsed_url = urlsplit(candidate_url)
                            if (
                                parsed_url.scheme not in {"http", "https"}
                                or not parsed_url.netloc
                            ):
                                raise ValueError(
                                    "URL must start with http:// or https:// "
                                    "and include a domain."
                                )

                            row_features = extract_features(candidate_url)
                            row_X = pd.DataFrame([row_features])

                            if hasattr(bulk_model, "feature_names_in_"):
                                row_X = row_X.reindex(
                                    columns=bulk_model.feature_names_in_,
                                    fill_value=0
                                )

                            row_prediction = int(bulk_model.predict(row_X)[0])
                            row_probability = None

                            if hasattr(bulk_model, "predict_proba"):
                                row_probabilities = bulk_model.predict_proba(row_X)[0]
                                row_classes = list(bulk_model.classes_)
                                if 1 in row_classes:
                                    row_probability = float(
                                        row_probabilities[row_classes.index(1)]
                                    )

                            row["prediction"] = row_prediction
                            row["result"] = (
                                "Phishing" if row_prediction == 1 else "Legitimate"
                            )
                            row["phishing_probability"] = (
                                round(row_probability, 6)
                                if row_probability is not None else None
                            )
                            row["risk_score"] = (
                                int(round(row_probability * 100))
                                if row_probability is not None else None
                            )
                            row["status"] = "Success"

                            save_scan_history(
                                candidate_url,
                                row_prediction,
                                row_probability
                            )

                        except Exception as row_error:
                            row["status"] = str(row_error)

                        bulk_rows.append(row)
                        progress.progress(
                            index / len(raw_urls),
                            text=f"Analysed {index} of {len(raw_urls)} URLs..."
                        )

                    progress.empty()
                    st.session_state["bulk_scan_results"] = pd.DataFrame(bulk_rows)

                    successful_count = sum(
                        item["status"] == "Success" for item in bulk_rows
                    )
                    phishing_count = sum(
                        item["result"] == "Phishing" for item in bulk_rows
                        if item["status"] == "Success"
                    )
                    st.success(
                        f"Bulk analysis finished: {successful_count} successful "
                        f"scans, {phishing_count} classified as phishing, and "
                        f"{len(bulk_rows) - successful_count} errors."
                    )

    except Exception as bulk_error:
        st.error(f"Could not read the uploaded CSV: {bulk_error}")

if "bulk_scan_results" in st.session_state:
    bulk_results_df = st.session_state["bulk_scan_results"]

    st.markdown("#### Bulk analysis results")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("URLs processed", len(bulk_results_df))
    metric_b.metric(
        "Phishing detected",
        int(((bulk_results_df["result"] == "Phishing") &
             (bulk_results_df["status"] == "Success")).sum())
    )
    metric_c.metric(
        "Errors",
        int((bulk_results_df["status"] != "Success").sum())
    )

    st.dataframe(
        bulk_results_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Bulk Scan Results (CSV)",
        data=bulk_results_df.to_csv(index=False).encode("utf-8"),
        file_name="surf_safe_bulk_scan_results.csv",
        mime="text/csv",
        key="download_bulk_scan_results"
    )



# --------------------------------------------------
# FEATURE 9: TYPOSQUATTING DETECTION
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 9 / LOOKALIKE DOMAIN SCREENING</div>
<div class="panel-heading">Typosquatting Detection</div>
<div class="panel-sub">
    Check whether a domain resembles a well-known brand through spelling
    changes or common digit substitutions. This check is local and does not
    visit the domain.
</div>
""", unsafe_allow_html=True)

typo_col, typo_button_col = st.columns([3, 1])
with typo_col:
    typo_input = st.text_input(
        "Domain or URL to check",
        placeholder="https://g00gle.com",
        key="typosquatting_input"
    )
with typo_button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    typo_clicked = st.button(
        "Check Lookalike",
        key="check_typosquatting",
        use_container_width=True
    )

if typo_clicked:
    if not typo_input.strip():
        st.warning("Enter a domain or URL to check.")
    else:
        try:
            typo_result = check_typosquatting(typo_input)
            st.session_state["typosquatting_result"] = typo_result

            if typo_result["is_lookalike"]:
                st.error(
                    f"Possible lookalike domain detected: "
                    f"`{typo_result['hostname']}` resembles "
                    f"`{typo_result['suggested_domain']}` "
                    f"({typo_result['similarity']:.0%} string similarity)."
                )
                st.warning(
                    "This is a spelling-similarity warning, not proof that "
                    "the domain is malicious. Do not enter credentials unless "
                    "you independently verify the exact domain."
                )
            else:
                st.success(
                    f"No close match to the built-in brand list was found "
                    f"for `{typo_result['hostname']}`."
                )
                st.caption(
                    "This does not mean the domain is safe. The local list is "
                    "limited and cannot identify every impersonation domain."
                )

        except ValueError as typo_error:
            st.warning(str(typo_error))

st.caption(
    "Screening list includes Google, Amazon, Microsoft, PayPal, Apple, "
    "Facebook, Instagram, WhatsApp, Netflix, Wikipedia, LinkedIn, Yahoo, "
    "Outlook, GitHub and Dropbox."
)



# --------------------------------------------------
# FEATURE 10: DOMAIN AGE CHECKER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 10 / DOMAIN REGISTRATION INTELLIGENCE</div>
<div class="panel-heading">Domain Age Checker</div>
<div class="panel-sub">
    Look up the domain's public registration record using RDAP. Newly
    registered domains can deserve extra scrutiny, but domain age alone
    cannot determine whether a website is malicious.
</div>
""", unsafe_allow_html=True)

age_col, age_button_col = st.columns([3, 1])
with age_col:
    age_input = st.text_input(
        "Domain or URL",
        placeholder="https://example.com",
        key="domain_age_input"
    )
with age_button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    age_clicked = st.button(
        "Check Domain Age",
        key="check_domain_age",
        use_container_width=True
    )

if age_clicked:
    if not age_input.strip():
        st.warning("Enter a domain or URL to check.")
    else:
        with st.spinner("Checking the public RDAP registration record..."):
            try:
                age_result = lookup_domain_registration(age_input)
                st.session_state["domain_age_result"] = age_result
            except ValueError as age_error:
                st.session_state.pop("domain_age_result", None)
                st.error(str(age_error))

if "domain_age_result" in st.session_state:
    age_result = st.session_state["domain_age_result"]

    st.markdown(f"#### Registration details: `{age_result['domain']}`")
    age_metric_1, age_metric_2, age_metric_3 = st.columns(3)
    age_metric_1.metric("Domain age", f"{age_result['age_days']:,} days")
    age_metric_2.metric("Approx. age", f"{age_result['age_years']:.1f} years")
    age_metric_3.metric(
        "Registration date",
        age_result["registration_date"].astimezone().strftime("%d %b %Y")
    )

    if age_result["expiration_date"] is not None:
        st.write(
            "**Expiration date:** "
            + age_result["expiration_date"].astimezone().strftime("%d %b %Y")
        )
    else:
        st.write("**Expiration date:** Not supplied in the RDAP record.")

    if age_result["age_days"] < 30:
        st.error(
            "Very recently registered domain (under 30 days). Treat this as "
            "an additional caution signal, not proof of phishing."
        )
    elif age_result["age_days"] < 90:
        st.warning(
            "Recently registered domain (under 90 days). Verify the site "
            "carefully before sharing information."
        )
    else:
        st.success(
            "The domain is at least 90 days old. Older registration does not "
            "guarantee that a website is safe."
        )

    st.caption(
        "Source: public RDAP registration data. This lookup sends the domain "
        "name to rdap.org and the domain registry's RDAP service. Some TLDs "
        "may not provide complete or public registration dates."
    )



# --------------------------------------------------
# FEATURE 11: REDIRECT CHAIN ANALYSIS
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 11 / URL REDIRECT INTELLIGENCE</div>
<div class="panel-heading">Redirect Chain Analysis</div>
<div class="panel-sub">
    Inspect where a URL redirects before reaching its final destination.
    SURF-Safe follows a limited number of redirects and blocks local or
    private-network destinations.
</div>
""", unsafe_allow_html=True)

redirect_col, redirect_button_col = st.columns([3, 1])
with redirect_col:
    redirect_input = st.text_input(
        "URL to inspect",
        placeholder="https://example.com",
        key="redirect_chain_input"
    )
with redirect_button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    redirect_clicked = st.button(
        "Analyze Redirects",
        key="analyze_redirect_chain",
        use_container_width=True
    )

if redirect_clicked:
    if not redirect_input.strip():
        st.warning("Enter a URL to inspect.")
    else:
        with st.spinner("Checking the URL and its redirect destinations..."):
            try:
                redirect_result = inspect_redirect_chain(redirect_input)
                st.session_state["redirect_chain_result"] = redirect_result
            except ValueError as redirect_error:
                st.session_state.pop("redirect_chain_result", None)
                st.error(str(redirect_error))
            except Exception as redirect_error:
                st.session_state.pop("redirect_chain_result", None)
                st.error(
                    "The redirect check could not be completed. "
                    f"Details: {redirect_error}"
                )

if "redirect_chain_result" in st.session_state:
    redirect_result = st.session_state["redirect_chain_result"]

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Redirects found", redirect_result["redirect_count"])
    metric_b.metric("Final HTTP status", redirect_result["status_code"])
    metric_c.metric(
        "Destination changed",
        "Yes" if redirect_result["redirect_count"] else "No"
    )

    st.write("**Original URL**")
    st.code(redirect_result["start_url"], language=None)
    st.write("**Final destination**")
    st.code(redirect_result["final_url"], language=None)

    if redirect_result["redirects"]:
        st.write("**Redirect chain**")
        for number, hop in enumerate(redirect_result["redirects"], start=1):
            st.markdown(
                f"**Hop {number} · HTTP {hop['status']}**  \n"
                f"`{hop['from']}`  \n"
                f"↓  \n"
                f"`{hop['to']}`"
            )
        st.warning(
            "A redirect is not automatically malicious. Review unexpected "
            "domain changes and continue checking the destination."
        )
    else:
        st.success("No HTTP redirects were observed.")

    st.caption(
        f"Content type: {redirect_result['content_type']}. "
        "This check makes an HTTP request to the supplied public URL. "
        "It does not submit forms or enter credentials. Redirect checks are "
        "best-effort and are not a substitute for a full security sandbox."
    )



# --------------------------------------------------
# FEATURE 12: SSL CERTIFICATE CHECKER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 12 / HTTPS CERTIFICATE INTELLIGENCE</div>
<div class="panel-heading">SSL Certificate Checker</div>
<div class="panel-sub">
    Inspect a public domain's TLS certificate, issuer, validity period,
    expiration date and negotiated TLS version. The checker validates the
    certificate chain and hostname and does not load the website's page.
</div>
""", unsafe_allow_html=True)

ssl_col, ssl_button_col = st.columns([3, 1])
with ssl_col:
    ssl_input = st.text_input(
        "Domain or URL to check",
        placeholder="https://example.com",
        key="ssl_certificate_input"
    )
with ssl_button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    ssl_clicked = st.button(
        "Check SSL Certificate",
        key="check_ssl_certificate",
        use_container_width=True
    )

if ssl_clicked:
    if not ssl_input.strip():
        st.warning("Enter a domain or HTTPS URL.")
    else:
        with st.spinner("Connecting securely and checking the TLS certificate..."):
            try:
                ssl_result = inspect_ssl_certificate(ssl_input)
                st.session_state["ssl_certificate_result"] = ssl_result
            except ValueError as ssl_error:
                st.session_state.pop("ssl_certificate_result", None)
                st.error(str(ssl_error))
            except (OSError, ssl.SSLError, TimeoutError) as ssl_error:
                st.session_state.pop("ssl_certificate_result", None)
                st.error(f"SSL certificate check failed: {ssl_error}")

if "ssl_certificate_result" in st.session_state:
    ssl_result = st.session_state["ssl_certificate_result"]

    st.markdown(f"#### Certificate details: `{ssl_result['hostname']}`")
    ssl_metric_1, ssl_metric_2, ssl_metric_3 = st.columns(3)
    ssl_metric_1.metric(
        "Certificate status",
        "Valid" if ssl_result["is_valid"] else "Not valid"
    )
    ssl_metric_2.metric(
        "Days until expiry",
        f"{ssl_result['days_remaining']:,}"
    )
    ssl_metric_3.metric("TLS version", ssl_result["tls_version"])

    st.write(f"**Certificate subject:** {ssl_result['subject']}")
    st.write(f"**Certificate issuer:** {ssl_result['issuer']}")
    if ssl_result["not_before"] is not None:
        st.write(
            "**Valid from:** "
            + ssl_result["not_before"].strftime("%d %b %Y %H:%M UTC")
        )
    st.write(
        "**Expiration date:** "
        + ssl_result["not_after"].strftime("%d %b %Y %H:%M UTC")
    )
    st.write(f"**Cipher:** `{ssl_result['cipher']}`")

    if ssl_result["san"]:
        with st.expander("Certificate DNS names (SAN)"):
            st.write(", ".join(ssl_result["san"]))

    if not ssl_result["is_valid"]:
        st.error(
            "The certificate is expired or not yet valid. Do not treat this "
            "connection as trusted."
        )
    elif ssl_result["days_remaining"] <= 14:
        st.warning(
            "The certificate is valid now but expires within 14 days."
        )
    elif ssl_result["days_remaining"] <= 30:
        st.warning(
            "The certificate is valid now but expires within 30 days."
        )
    else:
        st.success(
            "The certificate chain and hostname validation succeeded, and "
            "the certificate is currently within its validity period."
        )

    st.caption(
        "This check opens a TLS connection to the domain on port 443. "
        "It validates the certificate using the computer's trusted CA store. "
        "It does not inspect website content or prove that the website itself is safe."
    )




# --------------------------------------------------
# FEATURE 13: DNS SECURITY CHECKER
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 13 / DNS & EMAIL SECURITY INTELLIGENCE</div>
<div class="panel-heading">DNS Security Checker</div>
<div class="panel-sub">
    Inspect public DNS records for a domain, including A, AAAA, MX, NS and TXT.
    Check whether SPF and DMARC email-authentication records are published.
</div>
""", unsafe_allow_html=True)

dns_col, dns_button_col = st.columns([3, 1])
with dns_col:
    dns_input = st.text_input(
        "Domain or URL to inspect",
        placeholder="https://example.com",
        key="dns_security_input"
    )
with dns_button_col:
    st.markdown("<br>", unsafe_allow_html=True)
    dns_clicked = st.button(
        "Check DNS Security",
        key="check_dns_security",
        use_container_width=True
    )

if dns_clicked:
    if not dns_input.strip():
        st.warning("Enter a domain or URL to inspect.")
    else:
        with st.spinner("Querying public DNS records..."):
            try:
                dns_result = inspect_dns_security(dns_input)
                st.session_state["dns_security_result"] = dns_result
            except ValueError as dns_error:
                st.session_state.pop("dns_security_result", None)
                st.error(str(dns_error))

if "dns_security_result" in st.session_state:
    dns_result = st.session_state["dns_security_result"]
    dns_records = dns_result["records"]

    st.markdown(f"#### DNS results for `{dns_result['domain']}`")

    dns_metric_cols = st.columns(4)
    dns_metric_cols[0].metric(
        "IPv4 addresses (A)", len(dns_records["A"])
    )
    dns_metric_cols[1].metric(
        "IPv6 addresses (AAAA)", len(dns_records["AAAA"])
    )
    dns_metric_cols[2].metric(
        "Mail servers (MX)", len(dns_records["MX"])
    )
    dns_metric_cols[3].metric(
        "Nameservers (NS)", len(dns_records["NS"])
    )

    st.markdown("#### Email authentication checks")
    auth_cols = st.columns(2)

    with auth_cols[0]:
        if dns_result["has_spf"]:
            st.success("SPF record found")
            for record in dns_result["spf_records"]:
                st.code(record, language=None)
        else:
            st.warning("No SPF record was found in the domain's TXT records.")

    with auth_cols[1]:
        if dns_result["has_dmarc"]:
            st.success("DMARC record found")
            for record in dns_result["dmarc_records"]:
                st.code(record, language=None)
        else:
            st.warning("No DMARC record was found at _dmarc.")

    st.markdown("#### DNS record details")
    record_labels = {
        "A": "A records (IPv4)",
        "AAAA": "AAAA records (IPv6)",
        "MX": "MX records (mail servers)",
        "NS": "NS records (authoritative nameservers)",
        "TXT": "TXT records",
        "DMARC": "DMARC TXT lookup"
    }

    for record_key, record_title in record_labels.items():
        with st.expander(f"{record_title} ({len(dns_records[record_key])})"):
            if dns_records[record_key]:
                for record in dns_records[record_key]:
                    st.code(record, language=None)
            else:
                st.caption("No records returned.")

    if not dns_result["has_ns"]:
        st.warning("No NS records were returned. Check the domain spelling and DNS configuration.")

    st.caption(
        "Source: Cloudflare public DNS-over-HTTPS. The domain and DNS query "
        "names are sent to this DNS provider. Missing SPF or DMARC records "
        "are email-security findings, not proof that a website is phishing. "
        "This checker does not verify SPF/DMARC policy correctness or DNSSEC."
    )


# --------------------------------------------------
# FEATURE 15: COMBINED SECURITY RISK SCORE DASHBOARD
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; FEATURE 15 / COMBINED SECURITY INTELLIGENCE</div>
<div class="panel-heading">SURF-Safe Security Risk Score</div>
<div class="panel-sub">
    Combines the latest ML scan with supplementary checks that match the
    same registered domain. A higher score means more risk indicators were found.
</div>
""", unsafe_allow_html=True)

last_scan = st.session_state.get("sentinel_last_ml_scan")

if not last_scan:
    st.info("Run a URL scan above to generate the combined security risk score.")
else:
    scanned_url = last_scan["url"]
    scanned_domain_key = dashboard_domain_key(scanned_url)
    model_prediction = int(last_scan["prediction"])
    model_probability = last_scan.get("probability")

    # The ML model contributes up to 70 points. The remaining 30 points
    # are reserved for optional checks that were run against the same domain.
    if model_probability is not None:
        model_points = round(max(0.0, min(1.0, float(model_probability))) * 70)
        model_basis = f"ML phishing probability: {float(model_probability):.1%}"
    else:
        model_points = 70 if model_prediction == 1 else 10
        model_basis = (
            "ML classification only (the model does not provide probability); "
            "a conservative display weight is used."
        )

    components = [{
        "Check": "Machine-learning model",
        "Finding": model_basis,
        "Risk points": model_points,
        "Maximum": 70
    }]
    supplemental_points = 0
    checks_used = 0

    def same_dashboard_domain(candidate):
        return bool(candidate) and dashboard_domain_key(candidate) == scanned_domain_key

    # Typosquatting / brand-lookalike check.
    typo_data = st.session_state.get("typosquatting_result")
    if typo_data and same_dashboard_domain(typo_data.get("hostname")):
        checks_used += 1
        points = 15 if typo_data.get("is_lookalike") else 0
        supplemental_points += points
        finding = (
            f"Possible lookalike of {typo_data.get('suggested_domain')}"
            if points else "No close match in the built-in brand list"
        )
        components.append({
            "Check": "Typosquatting screening",
            "Finding": finding,
            "Risk points": points,
            "Maximum": 15
        })

    # Domain age check.
    age_data = st.session_state.get("domain_age_result")
    if age_data and same_dashboard_domain(age_data.get("domain")):
        checks_used += 1
        age_days = int(age_data.get("age_days", 0))
        points = 10 if age_days < 30 else 5 if age_days < 90 else 0
        supplemental_points += points
        components.append({
            "Check": "Domain registration age",
            "Finding": f"Registered {age_days:,} days ago",
            "Risk points": points,
            "Maximum": 10
        })

    # TLS certificate check.
    ssl_data = st.session_state.get("ssl_certificate_result")
    if ssl_data and same_dashboard_domain(ssl_data.get("hostname")):
        checks_used += 1
        if not ssl_data.get("is_valid", False):
            points = 10
            finding = "Certificate invalid or outside its validity period"
        elif int(ssl_data.get("days_remaining", 9999)) <= 14:
            points = 3
            finding = "Certificate expires within 14 days"
        else:
            points = 0
            finding = "Certificate validation passed"
        supplemental_points += points
        components.append({
            "Check": "TLS certificate",
            "Finding": finding,
            "Risk points": points,
            "Maximum": 10
        })

    # DNS email-authentication records. Missing records are low-weight
    # email-security observations, not proof that a website is phishing.
    dns_data = st.session_state.get("dns_security_result")
    if dns_data and same_dashboard_domain(dns_data.get("domain")):
        checks_used += 1
        points = (0 if dns_data.get("has_spf") else 2) + (
            0 if dns_data.get("has_dmarc") else 2
        )
        supplemental_points += points
        if dns_data.get("has_spf") and dns_data.get("has_dmarc"):
            finding = "SPF and DMARC records found"
        else:
            missing = []
            if not dns_data.get("has_spf"):
                missing.append("SPF")
            if not dns_data.get("has_dmarc"):
                missing.append("DMARC")
            finding = "Missing " + " and ".join(missing) + " record(s)"
        components.append({
            "Check": "Email DNS authentication",
            "Finding": finding,
            "Risk points": points,
            "Maximum": 4
        })

    # Redirect changes are a review signal only; redirects are common on
    # legitimate websites, so the score contribution is deliberately small.
    redirect_data = st.session_state.get("redirect_chain_result")
    if redirect_data and same_dashboard_domain(redirect_data.get("start_url")):
        checks_used += 1
        start_domain = dashboard_domain_key(redirect_data.get("start_url"))
        final_domain = dashboard_domain_key(redirect_data.get("final_url"))
        changed_domain = bool(final_domain and start_domain != final_domain)
        points = 5 if changed_domain else 0
        supplemental_points += points
        components.append({
            "Check": "Redirect destination",
            "Finding": (
                f"Redirected to a different domain: {final_domain}"
                if changed_domain else "No cross-domain redirect observed"
            ),
            "Risk points": points,
            "Maximum": 5
        })

    combined_score = min(100, max(0, model_points + supplemental_points))
    if combined_score < 25:
        score_label, score_color = "LOW", "#5aefb8"
    elif combined_score < 50:
        score_label, score_color = "MODERATE", "#ffd166"
    elif combined_score < 75:
        score_label, score_color = "HIGH", "#ff9f5a"
    else:
        score_label, score_color = "CRITICAL", "#ff8093"

    score_col, detail_col = st.columns([0.8, 1.7], gap="large")
    with score_col:
        st.markdown(f"""
        <div class="panel">
            <div class="section-kicker">OVERALL RISK / 100</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:58px;
                        font-weight:700;color:{score_color};line-height:1.15;">
                {combined_score}<span style="font-size:17px;color:#8d9bb2;"> / 100</span>
            </div>
            <div style="display:inline-block;margin-top:10px;padding:6px 12px;
                        border:1px solid {score_color};border-radius:20px;
                        color:{score_color};font-size:11px;font-weight:800;
                        letter-spacing:1px;">{score_label} RISK</div>
            <div style="color:#8d9bb2;font-size:11px;line-height:1.7;margin-top:13px;">
                Target: {html.escape(scanned_domain_key or scanned_url)}<br>
                ML scan: {html.escape(str(last_scan.get('timestamp', '')))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(combined_score / 100)

    with detail_col:
        st.markdown("**Risk contribution by check**")
        component_df = pd.DataFrame(components)
        st.dataframe(component_df, use_container_width=True, hide_index=True)
        st.caption(
            f"{checks_used} supplementary check(s) matched this domain. "
            "Checks for other domains are not included."
        )

    if checks_used == 0:
        st.info(
            "Only the ML model is included so far. To enrich this score, run "
            "the typosquatting, domain age, SSL, DNS and redirect checks using "
            "the same domain as the URL scanned above."
        )

    st.warning(
        "Educational risk indicator, not a certified security rating. "
        "The score uses a simple weighted heuristic, not a calibrated probability. "
        "Missing SPF/DMARC and redirects are weak context signals; neither alone "
        "proves phishing. A low score does not guarantee a website is safe."
    )


# --------------------------------------------------
# FEATURE 4: SCAN HISTORY
# --------------------------------------------------

st.markdown("---")
st.markdown("""
<div class="eyebrow">◈ &nbsp; ACTIVITY / LOCAL RECORDS</div>
<div class="panel-heading">Scan History</div>
<div class="panel-sub">
    Successful scans are saved on this computer in
    results/scan_history.csv.
</div>
""", unsafe_allow_html=True)

history_df = load_scan_history()

if history_df.empty:
    st.info("No scans have been saved yet. Analyze a URL to create your first history record.")
else:
    display_history = history_df.sort_values(
        by="timestamp", ascending=False
    ).head(20)

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="⬇ Download Complete Scan History (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="surf_safe_scan_history.csv",
        mime="text/csv",
        key="download_scan_history"
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">
    SURF-Safe &nbsp; • &nbsp; EXPLAINABLE PHISHING DETECTION
    <br><br>
    Built with Python, Streamlit & Machine Learning
    <br>
    For educational and research purposes
</div>
""", unsafe_allow_html=True)