"""paperlib.py -- small helpers to assemble the manuscript (.docx) with numbered citations and hyperlinks."""
import json, os, re
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
REFDIR = os.path.join(HERE, "..", "refs")
V = json.load(open(os.path.join(REFDIR, "refs_verified.json")))


def _auth(lst, maxn=6):
    lst = [a.strip() for a in lst if a.strip()]
    if len(lst) > maxn:
        return ", ".join(lst[:maxn]) + ", et al."
    return ", ".join(lst)


def _from_crossref(key, year=None, venue=None, extra=None, page=None, title=None, fix=None):
    r = V[key]
    authors = _auth(r["authors"])
    for old, new in (fix or {}).items():
        authors = authors.replace(old, new)
    s = f"{authors} ({year or r['year']}). {(title or r['title']).rstrip('.')}. "
    s += f"{venue or r['venue']}"
    if r.get("vol"):
        s += f", {r['vol']}"
        if r.get("issue") and r["issue"] not in ("0",):
            s += f"({r['issue']})"
    pg = page if page is not None else r.get("page")
    if pg:
        s += f", {pg.replace('-', '–')}"
    if extra:
        s += f", {extra}"
    return s + ".", "https://doi.org/" + r["doi"]


def _from_arxiv(aid, note=None, fix=None):
    r = V["arxiv:" + aid]
    authors = _auth(r["authors"])
    for old, new in (fix or {}).items():
        authors = authors.replace(old, new)
    s = f"{authors} ({r['year']}). {r['title'].rstrip('.')}. arXiv:{aid}"
    if note:
        s += f" ({note})"
    return s + ".", "https://arxiv.org/abs/" + aid


REFS = {}


def _add(key, pair):
    REFS[key] = pair


# ---- structural control, MR dampers ------------------------------------------------------------
_add("housner1997", _from_crossref("housner1997"))
_add("spencer2003", _from_crossref("spencer2003"))
_add("soong2002", _from_crossref("soong2002"))
_add("symans1999", _from_crossref("symans1999"))
_add("saaed2015", _from_crossref("saaed2015", year=2015))
_add("spencer1997", _from_crossref("spencer1997"))
_add("dyke1996", _from_crossref("dyke1996"))
_add("dyke1998", _from_crossref("dyke1998"))
_add("jansen2000", _from_crossref("jansen2000"))
_add("yi2001", _from_crossref("yi2001"))
_add("yang2002", _from_crossref("yang2002"))
_add("ohtori2004", _from_crossref("ohtori2004"))
_add("yoshida2004", _from_crossref("yoshida2004"))
_add("bitaraf2010", _from_crossref("bitaraf2010"))
_add("wen1976", _from_crossref("wen1976"))
_add("schurter2001", _from_crossref("schurter2001", venue="Proceedings of the 2001 American Control Conference", page="1097-1102"))
_add("banihani2006", _from_crossref("banihani2006", venue="Earthquake Engineering & Structural Dynamics"))
_add("chang2002", _from_crossref("chang2002"))
_add("dyke1996pdf", ("Dyke S.J., Spencer B.F. Jr., Sain M.K., Carlson J.D. (1996). Modeling and control of magnetorheological dampers "
                     "for seismic response reduction — author full text (PDF) hosted by the Smart Structures Technology Laboratory, "
                     "University of Illinois; used here to read Table 1, Table 2 and the structural matrices.",
                     "http://sstl.cee.illinois.edu/papers/MRD-SMS.pdf"))
_add("patrascu2012", ("Patrascu M. (2012). Hybrid geno-fuzzy controller for seismic vibration control. Proceedings of the UKACC "
                      "International Conference on Control 2012 (paper 255). Used here as a secondary source that lists the shear-mode "
                      "MR damper parameters attributed to Jansen and Dyke (2000).",
                      "https://skoge.folk.ntnu.no/prost/proceedings/ukaccc-2012/UKACC%20Control-Papers/010-Paper%20255.pdf"))
_add("chojung", ("Cho S.-W., Jung H.-J. (n.d.). Presentation slides (KAIST) containing a ‘System Data’ slide for the six-storey building "
                 "of Jansen and Dyke (2000). Used here only as a secondary source for the structural data.",
                 "https://www.slideserve.com/quinto/sang-won-cho-ph-d-candidate-kaist-hyung-jo-jung-research-assistant-professor-kaist"))
# ---- (deep) RL for structural control ---------------------------------------------------------
_add("khalat2019", _from_crossref("khalat2019", year=2019))
_add("rahmani2019", _from_crossref("rahmani2019"))
_add("eshkevari2021", _from_arxiv("2103.07616", fix={"Eshkevari S.S., Eshkevari S.S.": "Sadeghi Eshkevari S. (Soheila), Sadeghi Eshkevari S. (Soheil)"}))
_add("zhang2023", _from_crossref("zhang2023", page="", extra="Article 6770137"))
_add("kim2023", _from_crossref("kim2023"))
_add("hormozabad2024", _from_crossref("panda2024", venue="Practice Periodical on Structural Design and Construction"))
_add("gheni2024", _from_crossref("gheni2024", extra="Article 20220496"))
_add("thorat2025", _from_arxiv("2510.01269"))
_add("wang2026", _from_crossref("seismind"))
_add("asai2026", _from_crossref("asai2026", extra="Article 1244542"))
_add("kouts2026", _from_crossref("kouts2026", extra="Article 9603014", fix={"Caetano E.d.S.": "Caetano E.S."}))
_add("babaahmadi2023", _from_arxiv("2301.02714"))
# ---- RL methods -------------------------------------------------------------------------------
_add("sutton2018", ("Sutton R.S., Barto A.G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
                    "http://incompleteideas.net/book/the-book-2nd.html"))
_add("mnih2015", _from_crossref("mnih2015"))
_add("lillicrap2015", _from_arxiv("1509.02971"))
_add("schulman2017", _from_arxiv("1707.06347"))
_add("schulman2015", _from_arxiv("1506.02438"))
_add("fujimoto2018", _from_arxiv("1802.09477", fix={"Hoof H.v.": "van Hoof H."}))
_add("haarnoja2018", _from_arxiv("1801.01290"))
_add("kingma2014", _from_arxiv("1412.6980"))
_add("henderson2018", _from_crossref("henderson2018", venue="Proceedings of the AAAI Conference on Artificial Intelligence"))
_add("agarwal2021", _from_arxiv("2108.13264", "NeurIPS 2021"))
_add("kaelbling1998", _from_crossref("kaelbling1998"))
_add("tobin2017", _from_crossref("tobin2017", venue="2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)", page="23-30"))
_add("peng2018", _from_crossref("peng2018", venue="2018 IEEE International Conference on Robotics and Automation (ICRA)", page="3803-3810"))
_add("pinto2018", _from_crossref("pinto2018", venue="Robotics: Science and Systems XIV", page=""))
_add("openai2018", _from_arxiv("1808.00177"))
_add("rajeswaran2016", _from_arxiv("1610.01283"))
_add("muratore2022", _from_crossref("muratore2022", extra="Article 799893"))
_add("zhao2020", _from_arxiv("2009.13303"))
_add("banerjee2023", _from_arxiv("2309.01909"))
_add("raissi2019", _from_crossref("raissi2019"))
_add("karniadakis2021", _from_crossref("karniadakis2021"))
_add("kalman1960", _from_crossref("kalman1960"))
# ---- ground motions, statistics ---------------------------------------------------------------
_add("kanai1957", ("Kanai K. (1957). Semi-empirical formula for the seismic characteristics of the ground. Transactions of the "
                   "Architectural Institute of Japan, 57.1, 281–284.", "https://doi.org/10.3130/aijsaxx.57.1.0_281"))
_add("tajimi1958", ("Tajimi H. (1958). Estimation of the maximum response of building to earthquake motions by the statistical "
                    "method. Transactions of the Architectural Institute of Japan, 60.1, 249–252.",
                    "https://doi.org/10.3130/aijsaxx.60.1.0_249"))
_add("clough1975", ("Clough R.W., Penzien J. (1975). Dynamics of Structures. McGraw-Hill, New York (book; the link is the "
                    "published review in Journal of Applied Mechanics 44(2), 366)", "https://doi.org/10.1115/1.3424082"))
_add("rezaeian2008", _from_crossref("rezaeian2008", venue="Earthquake Engineering & Structural Dynamics"))
_add("vamvatsikos2002", _from_crossref("vamvatsikos2002", year=2002, venue="Earthquake Engineering & Structural Dynamics"))
_add("baker2015", _from_crossref("baker2015"))
_add("rockafellar2000", _from_crossref("rockafellar2000"))
_add("efron1979", _from_crossref("efron1979"))
_add("wilcoxon1945", _from_crossref("wilcoxon1945", page="80-83"))
_add("welch1947", _from_crossref("welch1947", title="The generalization of ‘Student's’ problem when several different population variances are involved", fix={"WELCH": "Welch"}))
_add("mann1947", _from_crossref("mann1947", title="On a test of whether one of two random variables is stochastically larger than the other"))
# ---- time delay in structural control (added with the delay-compensated baseline) ---------------
_add("cha2013", _from_crossref("cha2013", fix={"Cha Y.J.": "Cha Y.-J."}))
_r = _from_crossref("agrawal2000")
_add("agrawal2000", (_r[0], "https://doi.org/10.1002/(SICI)1096-9845(200001)29:1%3C37::AID-EQE894%3E3.0.CO;2-A"))
# ---- software and data ------------------------------------------------------------------------
_add("lam2015", _from_crossref("lam2015", title="Numba: a LLVM-based Python JIT compiler", venue="Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC", page="1-6"))
_add("paszke2019", _from_arxiv("1912.01703", "NeurIPS 2019"))
_add("gmspy", ("Yan Y. (2023). gmspy: Ground Motions Signal Processing for Python (version 0.1.3) [software, GPL licence]; source "
               "of ten of the recorded accelerograms used here (files bundled with the package, attributed there to the PEER "
               "strong-motion database).", "https://github.com/yexiang1992/gmspy"))
_add("vibrationdata", ("Irvine T. (n.d.). El Centro earthquake page (Vibrationdata): north–south component of the 18 May 1940 "
                       "Imperial Valley record, time in seconds and acceleration in g; source of the El Centro accelerogram used here.",
                       "https://www.vibrationdata.com/elcentro.htm"))
_add("peer", ("Pacific Earthquake Engineering Research Center (PEER). NGA-West2 ground-motion database.",
              "https://ngawest2.berkeley.edu/"))


# =================================================================================================

# ---------------------------------------------------------------------------------- OMML (Word equations)
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _m_run(t):
    italic = bool(re.search(r"[A-Za-zα-ωΑ-Ω]", t))
    sty = '<m:rPr><m:sty m:val="i"/></m:rPr>' if italic else '<m:rPr><m:sty m:val="p"/></m:rPr>'
    return ('<m:r>%s<m:t xml:space="preserve">%s</m:t></m:r>' % (sty, _esc(t)))


def _split_base(txt):
    """Split the trailing math atom (the base of a sub/superscript) off `txt`."""
    if not txt:
        return "", ""
    if txt.endswith(")"):
        d = 0
        for i in range(len(txt) - 1, -1, -1):
            if txt[i] == ")":
                d += 1
            elif txt[i] == "(":
                d -= 1
                if d == 0:
                    return txt[:i], txt[i:]
        return "", txt
    if txt.endswith("|"):
        j = txt.rfind("|", 0, len(txt) - 1)
        return (txt[:j], txt[j:]) if j >= 0 else ("", txt)   # base keeps the bars: |z|
    return txt[:-1], txt[-1:]


def _omml_body(text):
    """Convert text with ^{...} and _{...} markers into OMML runs with real sub/superscripts."""
    out, buf = [], ""
    toks = re.split(r"(\^\{.+?\}|_\{.+?\})", text)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t and not (t.startswith("^{") or t.startswith("_{")):
            buf += t
        elif t:
            kind, val = ("sup", t[2:-1]) if t.startswith("^{") else ("sub", t[2:-1])
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            other = ("sub" if kind == "sup" else "sup")
            paired = nxt.startswith("_{" if kind == "sup" else "^{")
            buf, base = _split_base(buf)
            if buf:
                out.append(_m_run(buf))
            if paired:
                v2 = nxt[2:-1]
                sub, sup = (val, v2) if kind == "sub" else (v2, val)
                out.append('<m:sSubSup><m:e>%s</m:e><m:sub>%s</m:sub><m:sup>%s</m:sup></m:sSubSup>'
                           % (_m_run(base), _m_run(sub), _m_run(sup)))
                i += 1
            elif kind == "sub":
                out.append('<m:sSub><m:e>%s</m:e><m:sub>%s</m:sub></m:sSub>' % (_m_run(base), _m_run(val)))
            else:
                out.append('<m:sSup><m:e>%s</m:e><m:sup>%s</m:sup></m:sSup>' % (_m_run(base), _m_run(val)))
            buf = ""
        i += 1
    if buf:
        out.append(_m_run(buf))
    return "".join(out)


def omml_math(text, display=False):
    """Return an lxml element holding the equation as native OMML math."""
    body = _omml_body(text)
    tag = "m:oMathPara" if display else "m:oMath"
    xml = '<%s xmlns:m="%s" xmlns:w="%s">%s</%s>' % (tag, M_NS, W_NS, body, tag)
    return parse_xml(xml)


class Paper:
    def __init__(self):
        self.doc = Document()
        self.order = []
        self.nfig = 0; self.ntab = 0; self.neq = 0
        sec = self.doc.sections[0]
        sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)          # A4
        sec.left_margin = sec.right_margin = Cm(2.0); sec.top_margin = sec.bottom_margin = Cm(2.3)
        st = self.doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(11)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        for i, sz in ((1, 14), (2, 12), (3, 11)):
            h = self.doc.styles[f"Heading {i}"]; h.font.name = "Times New Roman"; h.font.size = Pt(sz); h.font.bold = True
            h.font.color.rgb = RGBColor(0, 0, 0); h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # ---- citations
    def cite(self, *keys):
        nums = []
        for k in keys:
            if k not in REFS:
                raise KeyError(k)
            if k not in self.order:
                self.order.append(k)
            nums.append(self.order.index(k) + 1)
        nums = sorted(set(nums))
        # compress runs
        out = []; i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(str(nums[i]) if j == i else (f"{nums[i]}–{nums[j]}" if j > i + 1 else f"{nums[i]}, {nums[j]}"))
            i = j + 1
        return "[" + ", ".join(out) + "]"

    # ---- inline markup:  **bold**, *italic*, ^{sup}, _{sub}  (sub/superscripts may be nested inside bold/italic)
    def _runs(self, par, text, size=None):
        text = re.sub(r"(?<![\w\d.])-(?=\d)", "−", text)          # typographic minus for negative numbers
        # "1.9% in distribution, 1.9% out of distribution" -> "1.9% both in and out of distribution"
        text = re.sub(r"(\d+(?:\.\d+)?%) in distribution(?:,| and) \1 out of distribution", r"\1 both in and out of distribution", text)

        def emit(t, bold=False, italic=False):
            for u in re.split(r"(\^\{.+?\}|_\{.+?\})", t):
                if not u:
                    continue
                if u.startswith("^{"):
                    r = par.add_run(u[2:-1]); r.font.superscript = True
                elif u.startswith("_{"):
                    r = par.add_run(u[2:-1]); r.font.subscript = True
                else:
                    r = par.add_run(u)
                if bold:
                    r.bold = True
                if italic:
                    r.italic = True
                if size:
                    r.font.size = Pt(size)

        for t in re.split(r"(\*\*.+?\*\*|\*.+?\*)", text):
            if not t:
                continue
            if t.startswith("**"):
                emit(t[2:-2], bold=True)
            elif t.startswith("*"):
                emit(t[1:-1], italic=True)
            else:
                emit(t)
        return par

    def H(self, text, level=1):
        return self.doc.add_heading(text, level=level)

    def P(self, text, align="justify", size=None, space_after=6, indent=False):
        p = self.doc.add_paragraph()
        p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER, "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
        p.paragraph_format.space_after = Pt(space_after)
        if indent:
            p.paragraph_format.left_indent = Cm(0.6)
        return self._runs(p, text, size)

    def bullets(self, items):
        for it in items:
            p = self.doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(2)
            self._runs(p, it)

    def EQ(self, text):
        """Display equation as a native Word (OMML) equation object with a right-hand number."""
        self.neq += 1
        p = self.doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        p._p.append(omml_math(text))
        r = p.add_run("      (%d)" % self.neq); r.font.size = Pt(10); r.font.name = "Cambria Math"
        return self.neq

    def FIG(self, path, caption, width_cm=16.0):
        self.nfig += 1
        self.doc.add_picture(path, width=Cm(width_cm))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = self.P(f"**Figure {self.nfig}.** {caption}", size=9.5, space_after=10)
        return self.nfig

    def TAB(self, caption, header, rows, widths=None, size=8.5, bold_rows=()):
        self.ntab += 1
        self.P(f"**Table {self.ntab}.** {caption}", size=9.5, space_after=3, align="left")
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(header):
            c = t.rows[0].cells[j]; c.text = ""
            self._runs(c.paragraphs[0], f"**{h}**", size)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                c = t.rows[i + 1].cells[j]; c.text = ""
                s = str(v)
                self._runs(c.paragraphs[0], f"**{s}**" if i in bold_rows and s else s, size)
                for r in c.paragraphs[0].runs:
                    if i not in bold_rows:
                        r.bold = False
        if widths:
            t.autofit = False
            tblPr = t._tbl.tblPr
            for tag in ("w:tblW", "w:tblLayout"):
                for el in tblPr.findall(qn(tag)):
                    tblPr.remove(el)
            tw = OxmlElement("w:tblW"); tw.set(qn("w:w"), str(int(sum(widths) * 567))); tw.set(qn("w:type"), "dxa")
            lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed")
            tblPr.append(tw); tblPr.append(lay)
            grid = t._tbl.find(qn("w:tblGrid"))
            if grid is not None:
                for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
                    if j < len(widths):
                        gc.set(qn("w:w"), str(int(widths[j] * 567)))
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Cm(w)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)
        return self.ntab

    def _hyperlink(self, par, url, text):
        rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
        h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
        r = OxmlElement("w:r"); rpr = OxmlElement("w:rPr")
        c = OxmlElement("w:color"); c.set(qn("w:val"), "0563C1"); rpr.append(c)
        u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), "19"); rpr.append(sz)
        r.append(rpr)
        t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); r.append(t)
        h.append(r); par._p.append(h)

    def references(self):
        self.H("References", 1)
        for i, k in enumerate(self.order):
            txt, url = REFS[k]
            p = self.doc.add_paragraph(); p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.left_indent = Cm(0.8); p.paragraph_format.first_line_indent = Cm(-0.8)
            r = p.add_run(f"[{i + 1}] {txt} "); r.font.size = Pt(9.5)
            self._hyperlink(p, url, url)

    def references_md(self):
        return "\n".join(f"{i + 1}. {REFS[k][0]} <{REFS[k][1]}>" for i, k in enumerate(self.order))

    def save(self, path):
        self.doc.save(path)


if __name__ == "__main__":
    for k, (t, u) in REFS.items():
        print(k, "|", t[:150], "|", u)
