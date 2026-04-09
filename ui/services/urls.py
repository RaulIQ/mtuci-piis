import urllib.parse


def build_ws_url(api_url: str) -> str:
    parsed = urllib.parse.urlparse(api_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{ws_scheme}://{parsed.netloc}/ws/kws"
