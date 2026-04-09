DEFAULT_TARGET_LABELS = "one,two,three,four,five,six,seven,eight,nine"


def parse_target_labels(raw: str) -> list[str]:
    return [label.strip() for label in raw.split(",") if label.strip()]


def format_target_labels(raw: str) -> str:
    return ",".join(sorted(parse_target_labels(raw)))
