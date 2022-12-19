"""Creates the visualization of dependency distance"""
from pathlib import Path

import spacy

text = "The patient has experienced increasing suicidal thoughts"
#nlp = spacy.load("en_core_web_sm")

#doc = nlp(text)

#spacy.displacy.render(doc, style="dep")


ex = {
    "words": [
        {"text": "The", "tag": ""},
        {"text": "patient", "tag": ""},
        {"text": "has", "tag": ""},
        {"text": "experienced", "tag": ""},
        {"text": "increasing", "tag": ""},
        {"text": "suicidal", "tag": ""},
        {"text": "thoughts", "tag": ""},
    ],
    "arcs": [
        {"start": 0, "end": 1, "label": "1", "dir": "left"},
        {"start": 1, "end": 3, "label": "2", "dir": "left"},
        {"start": 2, "end": 3, "label": "1", "dir": "left"},
        {"start": 3, "end": 6, "label": "3", "dir": "right"},
        {"start": 4, "end": 6, "label": "2", "dir": "left"},
        {"start": 5, "end": 6, "label": "1", "dir": "left"},
    ],
}


opts = {"distance": 100, "word_spacing": 18, "arrow_width": 4}

svg = spacy.displacy.render(ex, style="dep", manual=True, jupyter=False, options=opts)
out_path = Path().cwd() / "dependency_visualization.svg"
out_path.open("w", encoding="utf-8").write(svg)
