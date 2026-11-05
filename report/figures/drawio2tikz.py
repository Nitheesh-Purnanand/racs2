import html
import re
import sys
import xml.etree.ElementTree as ET

SCALE = 0.0168


def style_dict(s):
    d = {}
    for part in (s or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
        elif part:
            d[part] = "1"
    return d


def to_latex(value):
    s = html.unescape(value or "")
    s = re.sub(r"<br\s*/?>", r" \\\\ ", s)
    s = re.sub(r"<i>([^<]+)</i><sub>([^<]+)</sub>", r"$\1_{\\mathrm{\2}}$", s)
    s = re.sub(r"<sub>([^<]+)</sub>", r"$_{\\mathrm{\1}}$", s)
    s = re.sub(r"<i>([^<]+)</i>", r"\\textit{\1}", s)
    s = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    for a, b in (("→", r"$\rightarrow$"), ("τ", r"$\tau$"), ("±", r"$\pm$"), ("≈", r"$\approx$"),
                 ("%", r"\%"), ("&", r"\&"), ("_{\\mathrm{τ}}", "")):
        s = s.replace(a, b)
    s = s.replace(r"\textit{$\tau$}", r"$\tau$")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main(src, dst, scale=SCALE):
    root = ET.parse(src).getroot()
    cells = root.findall(".//mxCell")
    geo = {}
    info = {}
    parent = {c.get("id"): c.get("parent") for c in cells}

    def absolute(cid):
        x, y = 0.0, 0.0
        while cid in geo:
            gx, gy, _, _ = geo[cid]
            x, y = x + gx, y + gy
            cid = parent.get(cid)
        return x, y

    for c in cells:
        if c.get("vertex") == "1":
            g = c.find("mxGeometry")
            geo[c.get("id")] = (float(g.get("x", 0)), float(g.get("y", 0)),
                                float(g.get("width")), float(g.get("height")))
            info[c.get("id")] = c

    boxes = {}
    for cid in geo:
        ax, ay = absolute(cid)
        _, _, w, h = geo[cid]
        boxes[cid] = (ax, ay, w, h)

    def P(x, y):
        return f"({x * scale:.3f},{-y * scale:.3f})"

    out = []
    headings = []
    for cid, c in info.items():
        st = style_dict(c.get("style"))
        if st.get("container") != "1":
            continue
        x, y, w, h = boxes[cid]
        out.append(f"\\draw[dashed, draw=black!70, rounded corners=2pt] {P(x, y)} rectangle {P(x + w, y + h)};")
        label = to_latex(c.get("value"))
        if st.get("verticalAlign") == "bottom":
            yy = y + h - 7
        else:
            yy = y + 8
        head = "fill=white, inner sep=1pt, font=\\diaglabel, text=black!75"
        if st.get("align") == "right":
            headings.append(f"\\node[anchor=east, {head}] at {P(x + w - 8, yy)} {{{label}}};")
        elif st.get("align") == "left":
            headings.append(f"\\node[anchor=west, {head}] at {P(x + 8, yy)} {{{label}}};")
        else:
            headings.append(f"\\node[{head}] at {P(x + w / 2, yy)} {{{label}}};")

    for cid, c in info.items():
        st = style_dict(c.get("style"))
        if st.get("container") == "1" or "text" in st:
            continue
        x, y, w, h = boxes[cid]
        arc = float(st.get("arcSize", 8)) / 100.0
        r = min(w, h) * arc * scale if arc < 0.5 else h / 2 * scale
        dashed = ", dashed" if st.get("dashed") == "1" else ""
        label = to_latex(c.get("value"))
        out.append(
            f"\\node[draw=black!75, fill=black!6, line width=0.6pt, rounded corners={r:.3f}cm{dashed}, "
            f"minimum width={w * scale:.3f}cm, minimum height={h * scale:.3f}cm, "
            f"text width={(w - 6) * scale:.3f}cm, align=center, inner sep=1pt, font=\\diagfont] "
            f"({cid}) at {P(x + w / 2, y + h / 2)} {{{label}}};")

    for cid, c in info.items():
        st = style_dict(c.get("style"))
        if "text" not in st:
            continue
        x, y, w, h = boxes[cid]
        out.append(f"\\node[fill=white, inner sep=1pt, font=\\diagfont] at {P(x + w / 2, y + h / 2)} "
                   f"{{{to_latex(c.get('value'))}}};")

    def port(bid, fx, fy):
        x, y, w, h = boxes[bid]
        return (x + fx * w, y + fy * h)

    for c in cells:
        if c.get("edge") != "1":
            continue
        st = style_dict(c.get("style"))
        g = c.find("mxGeometry")
        wps = [(float(p.get("x")), float(p.get("y")))
               for p in g.findall("./Array[@as='points']/mxPoint")]
        sp = g.find("./mxPoint[@as='sourcePoint']")
        tp = g.find("./mxPoint[@as='targetPoint']")
        src_id, dst_id = c.get("source"), c.get("target")
        if src_id in boxes:
            a = port(src_id, float(st.get("exitX", 0.5)), float(st.get("exitY", 1)))
            a_dir = "v" if st.get("exitY") in ("0", "1") else "h"
        else:
            a = (float(sp.get("x")), float(sp.get("y")))
            a_dir = "v"
        if dst_id in boxes:
            b = port(dst_id, float(st.get("entryX", 0.5)), float(st.get("entryY", 0)))
            b_dir = "v" if st.get("entryY") in ("0", "1") else "h"
        else:
            b = (float(tp.get("x")), float(tp.get("y")))
            b_dir = "v"
        pts = [a] + wps + [b]
        route = [pts[0]]
        for i in range(1, len(pts)):
            p, q = route[-1], pts[i]
            if abs(p[0] - q[0]) > 0.5 and abs(p[1] - q[1]) > 0.5:
                first = i == 1
                last = i == len(pts) - 1
                if first and last and a_dir == "v" and b_dir == "v":
                    my = (p[1] + q[1]) / 2
                    route += [(p[0], my), (q[0], my)]
                elif first and a_dir == "v":
                    route.append((p[0], q[1]))
                elif first and a_dir == "h":
                    route.append((q[0], p[1]))
                elif last and b_dir == "v":
                    route.append((q[0], p[1]))
                else:
                    route.append((p[0], q[1]))
            route.append(q)
        clean = [route[0]]
        for q in route[1:]:
            if abs(q[0] - clean[-1][0]) > 0.01 or abs(q[1] - clean[-1][1]) > 0.01:
                clean.append(q)
        path = " -- ".join(P(*q) for q in clean)
        out.append(f"\\draw[-{{Stealth[length=1.8mm]}}, line width=0.6pt] {path};")

    out += headings
    body = "\n\t".join(out)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("% Generated from " + src.replace("\\", "/").split("/")[-1]
                + " by drawio2tikz.py; edit the .drawio file and rerun.\n")
        f.write("\\begin{tikzpicture}\n\t\\hyphenpenalty=10000\\exhyphenpenalty=10000\n\t"
                + body + "\n\\end{tikzpicture}\n")
    print("wrote", dst, len(out), "items")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else SCALE)
