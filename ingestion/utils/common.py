from dataclasses import dataclass


@dataclass
class CustomDocument:
    file_path: str
    title: str
    content: str
    short_description: str
    page_number: int
    page_count: int
    metadata: dict
