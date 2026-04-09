import urllib.parse


def build_ws_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/ws/kws"


def build_ws_kws_logmel_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/ws/kws-logmel"
