from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class OCRVector:
    x: float
    y: float

    def __iter__(self):
        return iter((self.x, self.y))


@dataclass
class OCRBoundingBox:
    normalizedVertices: List

    def __init__(self, normalizedVertices):
        self.normalizedVertices = [OCRVector(vertex["x"], vertex["y"]) for vertex in normalizedVertices]

    def __getitem__(self, item):
        return self.normalizedVertices[item]


break_types = {
    "HYPHEN": "-",
    "LINE_BREAK": "\n",
    "SPACE": " ",
    "EOL_SURE_SPACE": "\n",
}


@dataclass
class OCRSymbol:
    confidence: Optional[float]
    text: str
    property: Optional[Dict]

    def get_break(self):
        if self.property is not None:
            break_type = self.property['detectedBreak']['type']
            if break_type not in break_types:
                breakpoint()
            return break_types[break_type]
        return ""
    pass


@dataclass
class OCRWord:
    boundingBox: OCRBoundingBox
    confidence: float
    symbols: List[OCRSymbol]
    property: Optional[Dict]

    def get_text(self):
        symbol_list = []
        for symbol in self.symbols:
            if symbol.property is not None and (len(symbol.property) != 1 or "detectedBreak" not in symbol.property):
                breakpoint()
            symbol_list.append(f"{symbol.text}{symbol.get_break()}")
        return "".join(symbol_list)
    pass


@dataclass
class OCRParagraph:
    boundingBox: OCRBoundingBox
    confidence: float
    words: List[OCRWord]

    def get_text(self):
        return " ".join(word.get_text() for word in self.words)

    pass


@dataclass
class OCRBlock:
    boundingBox: OCRBoundingBox
    paragraphs: List[OCRParagraph]
    blockType: str
    confidence: float

    def get_text(self):
        return "\n".join(paragraphs.get_text() for paragraphs in self.paragraphs)


@dataclass
class OCRPage:
    page_num: int
    confidence: float
    height: int
    width: int
    languages: List
    blocks: List[OCRBlock]


def extract_data_structure(data):
    structure = {}

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict) or isinstance(value, list):
                structure[key] = extract_data_structure(value)
            else:
                structure[key] = type(value).__name__
    elif isinstance(data, list):
        if data:
            structure = [extract_data_structure(data[0])]
        else:
            structure = ['Optional']

    return structure


def merge_key_caches(d1, d2):
    merged_keys = list(set(list(d1.keys()) + list(d2.keys())))
    return {key: d1.get(key, set()).union(d2.get(key, set())) for key in merged_keys}


def extract_structure(data):
    page_fields = set()
    block_fields = set()
    paragraph_fields = set()
    word_fields = set()
    symbol_fields = set()
    word_property_fields = set()
    symbol_property_fields = set()
    for page in data:
        page_fields = page_fields.union(set(page.keys()))
        for block in page["blocks"]:
            block_fields = block_fields.union(set(block.keys()))
            for paragraph in block["paragraphs"]:
                paragraph_fields = paragraph_fields.union(set(paragraph.keys()))
                for word in paragraph["words"]:
                    word_fields = word_fields.union(set(word.keys()))
                    for symbol in word["symbols"]:
                        for property_key, property_value in symbol.get("property", {}).items():
                            symbol_property_fields.add(property_key)
                        symbol_fields = symbol_fields.union(set(symbol.keys()))
                    for property_key, property_value in word.get("property", {}).items():
                        word_property_fields.add(property_key)
                        pass
    return {
        "page_fields": page_fields,
        "block_fields": block_fields,
        "paragraph_fields": paragraph_fields,
        "word_fields": word_fields,
        "symbol_fields": symbol_fields,
        "word_property_fields": word_property_fields,
        "symbol_property_fields": symbol_property_fields,
    }


if __name__ == '__main__':
    pass
