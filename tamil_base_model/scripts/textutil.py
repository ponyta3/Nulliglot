"""Shared helpers for cleaning the Tamil sources."""
import html, re
from html.parser import HTMLParser

TA = re.compile(r"[஀-௿]")
TAW = re.compile(r"[஀-௿]+")
# letter sequences that cannot occur in Tamil: a word starting with a vowel sign or pulli, a
# vowel sign after a vowel sign or an independent vowel, two pulli in a row
BADSHAPE = re.compile(r"^[ா-்ௗ]|[ா-ௌௗ][ா-்]|[அ-ஔ][ா-்]|்்")
BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "tr",
         "table", "section", "article", "hr", "dd", "dt", "figure", "figcaption", "title", "body"}
SKIP = {"script", "style", "pre", "head", "nav", "svg", "math", "figcaption", "iframe", "form"}

def nwords(s):
    return len(TAW.findall(s))

def tidy(s):
    s = s.replace("‌", "").replace("‍", "").replace("­", "").replace("﻿", "")
    return re.sub(r"[ \t  -​　]+", " ", s).strip()

OTHERW = re.compile(r"[^\W\d_\u0B80-\u0BFF]+")       # words of any other script

def is_tamil(par, share=0.9):
    """True when the paragraph is Tamil text: at least `share` of its words are in Tamil script
    (so a few English names or terms are allowed), it has at least 3 Tamil letters, and Tamil
    letters are at least half of all non-space characters (this drops number tables)."""
    ta = len(TAW.findall(par)); other = len(OTHERW.findall(par)); letters = len(TA.findall(par))
    return letters >= 3 and ta >= share * (ta + other) and letters >= 0.5 * len(par.replace(" ", ""))

def other_share(text):
    """Share of words that are not in Tamil script."""
    ta = len(TAW.findall(text)); other = len(OTHERW.findall(text))
    return other / (ta + other) if ta + other else 0.0

def badshape_share(text):
    w = TAW.findall(text)
    return sum(1 for x in w if BADSHAPE.search(x)) / len(w) if w else 0.0

class _Text(HTMLParser):
    def __init__(self, skip_classes=(), skip_tables=False):
        super().__init__(convert_charrefs=True)
        self.out = []; self.skip = 0; self.stack = []; self.skip_classes = skip_classes
        self.skip_tags = SKIP | ({"table"} if skip_tables else set())
    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        bad = tag in self.skip_tags or any(c in cls for c in self.skip_classes)
        if tag not in ("br", "hr", "img", "meta", "link", "input"):
            self.stack.append((tag, bad)); self.skip += bad
        if tag in BLOCK: self.out.append("\n")
    def handle_endtag(self, tag):
        if not any(t == tag for t, _ in self.stack):       # stray end tag
            if tag in BLOCK: self.out.append("\n")
            return
        while self.stack:
            t, bad = self.stack.pop(); self.skip -= bad
            if t == tag: break
        if tag in BLOCK: self.out.append("\n")
    def handle_data(self, d):
        if not self.skip: self.out.append(d)

def html_paragraphs(markup, skip_classes=(), skip_tables=False):
    """Text of an HTML fragment as a list of paragraphs (block elements and <br> end one)."""
    p = _Text(skip_classes, skip_tables)
    try:
        p.feed(markup); p.close()
    except Exception:
        pass
    return [t for t in (tidy(x) for x in "".join(p.out).split("\n")) if t]
