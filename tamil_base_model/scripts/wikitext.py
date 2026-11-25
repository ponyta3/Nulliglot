"""Turn MediaWiki markup into plain text. Used for Tamil Wikisource, Wikinews, Wikibooks, Wikiquote.

strip(text, counts) returns plain text in which paragraphs are separated by a blank line and
the lines of a poem by a single newline. Templates are handled by name:
  KEEP_LAST   formatting wrappers: the wrapped text is kept
  BLOCK       wrappers that also start a new paragraph (centred headings, quote boxes)
  special     a few with their own rule (hyphenated words, initials, gaps, breaks)
  all others  dropped. `counts` (a Counter) records every dropped template that contained
              Tamil words, so that nothing is lost unnoticed.
"""
import bz2, html, re
import xml.etree.ElementTree as ET

TAW = re.compile(r"[஀-௿]+")
NL = "\x01"            # a line break that must survive (poem line, <br>)
PB = "\n\n"

KEEP_LAST = {
    "larger", "x-larger", "xx-larger", "xxx-larger", "xxxx-larger", "x–larger", "smaller", "x-smaller",
    "xx-smaller", "fine", "small", "big", "bold", "b", "i", "u", "em", "sc", "fs", "font-size", "red",
    "darkred", "left margin", "hanging indent", "hi", "indent", "text-indent",
    "letter-spacing", "justify", "unicode", "lang", "nowrap", "float left", "left",
    "வலச்சிறிது", "முக்கால்சிறி", "border-radius", "background color", "dual line",
}
BLOCK = {
    "center", "c", "centre", "block center", "bc", "center block", "centered box", "box", "border",
    "block left", "block indent", "cquote", "quote", "quote box", "blockquote", "dialogue indented",
    "ppoem", "நடுதடி", "தலைப்பு-நடு-தடிப்பு",
}
COLOURS = {"blue", "green", "maroon", "teal", "color", "colour", "gray", "grey", "black", "brown", "orange",
           "purple", "pink", "polytonic", "ellipsed text"}
INVISIBLE = {"white", "hidden", "invisible"}
RIGHT = {"right", "float right", "block right", "r", "வலது"}
FIRST = {"sic", "tooltip", "largeinitial", "dropinitial", "di", "reconstruct"}
BREAK = {"nop", "nopt", "dhr", "***", "rule", "custom rule", "bar", "clear", "கோடு", "psm rule", "-"}
SPACE = {"gap", "gap+", "gap2", "spaces", "space", "ndash", "–", "em dash"}
HYPH_START = {"hws", "hyphenated word start"}
HYPH_END = {"hwe", "hyphenated word end"}

KNOWN = KEEP_LAST | BLOCK | RIGHT | FIRST | BREAK | SPACE | HYPH_START | HYPH_END | {
    "fqm", "sfrac", "sfrac nobar", "!", "block center/s", "block center/e", "border/s", "border/e",
    "hanging indent/s", "hanging indent/e", "hanging indent/m", "multicol", "multicol-break", "multicol-end",
    "columns", "css image crop", "raw image", "rh", "dtpl", "tlr", "dlr", "djvu page link", "djvu page link 2",
    "dotted toc page listing", "toc page listing", "ts", "cs", "anchor", "reflist", "smallrefs", "sup", "sub",
    "runningheader/core", "table", "familytree", "rotate", "wdl", "pline", "brace", "brace2", "overfloat left",
    "overfloat image"}

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_DROP_TAGS = re.compile(r"<(noinclude|ref|math|gallery|source|syntaxhighlight|pre|code|timeline|score|"
                        r"imagemap|templatestyles|nowiki|chem|hiero|graph|mapframe|score)\b[^>]*?(/>|>.*?</\1\s*>)", re.S | re.I)
_SELF_REF = re.compile(r"<(ref|references|section|pagequality|templatestyles)\b[^>]*/?>", re.I)
_POEM = re.compile(r"<poem\b[^>]*>(.*?)</poem\s*>", re.S | re.I)
_TABLE = re.compile(r"\{\|(?:(?!\{\||\|\}).)*\|\}", re.S)
_TPL = re.compile(r"\{\{((?:(?!\{\{|\}\}).)*)\}\}", re.S)
_LINK = re.compile(r"\[\[((?:(?!\[\[|\]\]).)*)\]\]", re.S)
_EXT = re.compile(r"\[(?:https?:|ftp:|//)[^\s\]]*\s*([^\]]*)\]")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9]*\b[^<>]*>")
_NAMED = re.compile(r"^\s*[\w .-]{1,25}=", re.A)
_NS = re.compile(r"^\s*:?\s*(file|image|படிமம்|கோப்பு|category|பகுப்பு|media|[a-z-]{2,12})\s*:", re.I)
_TEXT_NS = re.compile(r"^\s*:?\s*(w|wikt|s|q|b|n|wikipedia|wiktionary|விக்கிப்பீடியா|ஆசிரியர்|author|பக்கம்|page|"
                      r"அட்டவணை|index|வலைவாசல்|portal)\s*:", re.I)

def _link(m):
    body = m.group(1)
    parts = body.split("|")
    if _TEXT_NS.match(parts[0]):
        return parts[-1] if len(parts) > 1 else parts[0].split(":", 1)[1]
    if _NS.match(parts[0]):
        return ""                                   # files, categories, interlanguage links
    return parts[-1] if len(parts) > 1 else parts[0].split("#")[0]

def _template(m, counts):
    parts = m.group(1).split("|")
    name = re.sub(r"[\s_]+", " ", parts[0].lower().replace("\u200e", "")).strip()
    pos = [p for p in parts[1:] if not _NAMED.match(p)]
    if name in INVISIBLE: return ""
    if name not in KNOWN:                           # misspelt or rarer wrappers
        if name in COLOURS or "larger" in name or "smaller" in name or "underline" in name: name = "larger"
        elif name.endswith(" block") or "center" in name or "centr" in name or name.startswith("block"): name = "center"
    if name in KEEP_LAST:
        return pos[-1].strip() if pos else ""
    if name in BLOCK:
        return PB + pos[-1].strip() + PB if pos else PB
    if name in RIGHT:                               # right-aligned: a signature, or a verse number
        t = pos[-1].strip() if pos else ""
        return PB + t + PB if TAW.search(t) else " "
    if name in FIRST:
        return pos[0].strip() if pos else ""
    if name in BREAK: return PB
    if name in SPACE: return " "
    if name in HYPH_START:                          # word split over two pages: whole word here
        return (pos[1] if len(pos) > 1 else pos[0] if pos else "").strip()
    if name in HYPH_END: return ""
    if name == "fqm": return pos[0].strip() if pos else "“"
    if name in ("sfrac", "sfrac nobar") and len(pos) >= 2: return "/".join(p.strip() for p in pos[-2:])
    if name == "!": return "|"
    if name in ("block center/s", "block center/e", "border/s", "border/e",
                "hanging indent/s", "hanging indent/e", "hanging indent/m", "multicol", "multicol-break",
                "multicol-end", "columns"):
        return PB
    if counts is not None:
        lost = len(TAW.findall(" ".join(pos)))
        if lost >= 3: counts[name] += lost
    return ""

def strip(text, counts=None, drop_tables=True):
    t = _COMMENT.sub("", text)
    t = _DROP_TAGS.sub(" ", t)
    t = _SELF_REF.sub("", t)
    # poems: every line break counts
    t = _POEM.sub(lambda m: PB + PB.join(st.strip("\n").replace("\n", NL) for st in
                                          re.split(r"\n[ \t]*\n", m.group(1).strip("\n"))) + PB, t)
    t = re.sub(r"<br\s*/?>", NL, t, flags=re.I)
    for _ in range(8):                               # links, innermost first
        t, n = _LINK.subn(_link, t)
        if not n: break
    for _ in range(12):                              # templates, innermost first
        t, n = _TPL.subn(lambda m: _template(m, counts), t)
        if not n: break
    if drop_tables:
        for _ in range(4):
            t, n = _TABLE.subn(PB, t)
            if not n: break
    t = _EXT.sub(lambda m: m.group(1), t)
    t = _URL.sub("", t)
    t = re.sub(r"</?(p|div|center|blockquote|li|ul|ol|table|tr|td|th|h[1-6]|hr|dd|dt|dl)\b[^<>]*>", PB, t, flags=re.I)
    t = _TAG.sub("", t)
    t = html.unescape(t).replace(" ", " ")
    t = re.sub(r"'{2,5}", "", t)
    t = re.sub(r"__[A-Z]+__", "", t)
    t = re.sub(r"\{\{|\}\}|\[\[|\]\]", "", t)        # unbalanced leftovers
    out = []
    for block in re.split(r"\n\s*\n", t):
        lines = []
        for l in block.split("\n"):
            h = re.match(r"^\s*(={1,6})\s*(.*?)\s*\1\s*$", l)
            if h: lines.append(PB + h.group(2) + PB); continue
            if re.match(r"^\s*-{4,}\s*$", l): lines.append(PB); continue
            if re.match(r"^\s*(\{\||\|\}|\|-|[|!])", l): lines.append(PB); continue   # stray table rows
            b = re.match(r"^[*#:;]+\s*(.*)$", l)
            lines.append(PB + b.group(1) + PB if b else l)
        out.append(" ".join(lines))
    t = PB.join(out)
    paras = []
    for p in re.split(r"\n\s*\n", t):
        p = "\n".join(x for x in (re.sub(r"[ \t]+", " ", l).strip() for l in p.replace("\n", " ").split(NL)) if x)
        if p: paras.append(p)
    return paras


def dump_pages(path):
    """Yield (page id, title, namespace, wikitext, is_redirect) from a pages-articles dump."""
    for _, el in ET.iterparse(bz2.open(path, "rb")):
        if el.tag.endswith("}page"):
            tx = el.find("./{*}revision/{*}text")
            yield (el.find("./{*}id").text, el.find("./{*}title").text, int(el.find("./{*}ns").text),
                   (tx.text or "") if tx is not None else "", el.find("./{*}redirect") is not None)
            el.clear()
