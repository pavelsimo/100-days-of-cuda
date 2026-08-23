#!/usr/bin/env python3
"""Static site generator for the 100-days-of-cuda GitHub Pages site.

Renders README.md and the PMPP chapter notes to HTML, copies the
interactive animations and images, and writes everything to _site/.

Usage: python scripts/build_site.py
Deps:  pip install markdown pymdown-extensions pygments mdx-truly-sane-lists
"""

import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
ASSETS = Path(__file__).resolve().parent / "assets"

REPO_URL = "https://github.com/pavelsimo/100-days-of-cuda"
BLOB_URL = f"{REPO_URL}/blob/main/"

PMPP_DIR = ROOT / "docs" / "programming-massively-parallel-processors"
ANIM_DIR = ROOT / "docs" / "animations"
IMAGE_EXTS = {".png", ".gif", ".jpg", ".jpeg", ".svg", ".webp"}

MD_EXTENSIONS = [
    "tables",
    "mdx_truly_sane_lists",  # GitHub-style 2-space nested lists
    "toc",
    "pymdownx.superfences",
    "pymdownx.highlight",
    "pymdownx.arithmatex",
]
MD_CONFIG = {
    "mdx_truly_sane_lists": {"nested_indent": 2},
    "pymdownx.highlight": {"css_class": "highlight", "guess_lang": False},
    "pymdownx.arithmatex": {"generic": True},
    "toc": {"permalink": False},
}

KATEX_HEAD = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[
    {left:'\\\\[',right:'\\\\]',display:true},
    {left:'\\\\(',right:'\\\\)',display:false}]})"></script>
"""

# Injected into the copied animation pages so visitors can get back to the
# site. Uses the animation's own CSS variables, so it follows its theme toggle.
BACK_NAV_STYLE = """<style>
.site-back{position:fixed;top:12px;left:12px;z-index:100;display:flex;gap:6px}
.site-back a{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:10.5px;letter-spacing:.18em;text-transform:uppercase;
  padding:7px 14px;border:1px solid var(--panel-edge);
  color:var(--txt-dim);background:var(--panel);text-decoration:none;
  clip-path:polygon(8px 0,100% 0,calc(100% - 8px) 100%,0 100%);
  transition:all .25s;
}
.site-back a:hover{color:var(--nv-green-bright);border-color:var(--nv-green-dim)}
@media (max-width:760px){.site-back{position:static;justify-content:center;margin-bottom:14px}}
</style>"""
BACK_NAV_HTML = """<nav class="site-back">
  <a href="../index.html">&lsaquo; Home</a>
  <a href="index.html">Animations</a>
</nav>"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script>try{{if(localStorage.getItem('theme')==='light')document.documentElement.classList.add('light')}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="{root}assets/style.css">{extra_head}
</head>
<body>

<header class="site">
  <h1 class="site-title"><a href="{root}index.html">100 Days of <span class="accent">CUDA</span></a></h1>
  <div class="subtitle">{subtitle}</div>
</header>

<nav class="pipeline">
  <a href="{root}index.html"{active_home}>Home</a>
  <a href="{root}pmpp/index.html"{active_pmpp}>PMPP Notes</a>
  <a href="{root}animations/index.html"{active_anim}>Animations</a>
  <a href="{repo_url}">GitHub</a>
  <button id="btnTheme" type="button">&#9675; Theme</button>
</nav>

{content}

<footer class="site">
  <a href="{repo_url}">pavelsimo/100-days-of-cuda</a> &middot; built with nvcc-flavored markdown
</footer>

<script>
(() => {{
  const btn = document.getElementById('btnTheme');
  const sync = () => btn.textContent =
    document.documentElement.classList.contains('light') ? '\\u25CF Dark' : '\\u25CB Light';
  btn.onclick = () => {{
    const light = document.documentElement.classList.toggle('light');
    try {{ localStorage.setItem('theme', light ? 'light' : 'dark'); }} catch (e) {{}}
    sync();
  }};
  sync();
}})();
</script>
</body>
</html>
"""


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=MD_EXTENSIONS, extension_configs=MD_CONFIG)


def render_page(*, title, subtitle, content, root="", active="", math=False) -> str:
    return PAGE_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        content=content,
        root=root,
        repo_url=REPO_URL,
        extra_head=KATEX_HEAD if math else "",
        active_home=' class="active"' if active == "home" else "",
        active_pmpp=' class="active"' if active == "pmpp" else "",
        active_anim=' class="active"' if active == "anim" else "",
    )


def rewrite_readme_links(html: str) -> str:
    """Point repo-relative links at the right place on the site (or GitHub)."""

    def repl(m):
        attr, url = m.group(1), m.group(2)
        if url.startswith(("http://", "https://", "#", "mailto:")):
            return m.group(0)
        if url.startswith("docs/animations/"):
            new = "animations/" + url.rsplit("/", 1)[-1]
        elif url.startswith("docs/programming-massively-parallel-processors/"):
            new = "pmpp/" + url.rsplit("/", 1)[-1].replace(".md", ".html")
        elif url.startswith("images/"):
            new = url
        else:
            new = BLOB_URL + url
        return f'{attr}="{new}"'

    return re.sub(r'(href|src)="([^"]+)"', repl, html)


def collect_chapters():
    chapters = []
    for path in PMPP_DIR.glob("chapter-*.md"):
        num = int(re.search(r"chapter-(\d+)", path.name).group(1))
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^# .*Chapter\s+\d+[,:]?\s*(.*)$", text, re.M)
        title = m.group(1).strip() if m else path.stem
        chapters.append({"num": num, "title": title, "path": path, "text": text})
    chapters.sort(key=lambda c: c["num"])
    return chapters


def collect_animations():
    anims = []
    for path in sorted(ANIM_DIR.glob("*.html")):
        m = re.search(r"<title>(.*?)</title>", path.read_text(encoding="utf-8"))
        title = m.group(1).strip() if m else path.stem
        anims.append({"title": title, "file": path.name, "path": path})
    return anims


def chapter_list_html(chapters, prefix=""):
    items = "\n".join(
        f'<li><a href="{prefix}chapter-{c["num"]}.html">'
        f'<span class="tag">CH{c["num"]:02d}</span>{c["title"]}</a></li>'
        for c in chapters
    )
    return f'<ul class="link-list">\n{items}\n</ul>'


def animation_list_html(anims, prefix=""):
    items = "\n".join(
        f'<li><a href="{prefix}{a["file"]}">'
        f'<span class="tag">&#9654;</span>{a["title"]} '
        f'<span class="dim">&middot; interactive</span></a></li>'
        for a in anims
    )
    return f'<ul class="link-list">\n{items}\n</ul>'


def build():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    (OUT / "pmpp").mkdir()
    (OUT / "animations").mkdir()

    shutil.copytree(ASSETS, OUT / "assets")
    (OUT / "images").mkdir()
    for img in (ROOT / "images").iterdir():
        if img.suffix.lower() in IMAGE_EXTS:
            shutil.copy2(img, OUT / "images" / img.name)
    (OUT / ".nojekyll").write_text("")

    chapters = collect_chapters()
    anims = collect_animations()
    for a in anims:
        text = a["path"].read_text(encoding="utf-8")
        text = text.replace("</head>", BACK_NAV_STYLE + "\n</head>", 1)
        text = text.replace("<body>", "<body>\n" + BACK_NAV_HTML, 1)
        (OUT / "animations" / a["file"]).write_text(text, encoding="utf-8")

    # ---- index.html: rendered README ----
    readme_html = rewrite_readme_links(md_to_html((ROOT / "README.md").read_text(encoding="utf-8")))
    sections = f"""
<article class="panel">
  <div class="panel-title">Progress Log &middot; README.md</div>
  <div class="md">
{readme_html}
  </div>
</article>
"""
    (OUT / "index.html").write_text(
        render_page(
            title="100 Days of CUDA",
            subtitle="one kernel at a time &middot; day by day progress log",
            content=sections,
            active="home",
        ),
        encoding="utf-8",
    )

    # ---- pmpp/index.html ----
    pmpp_index = f"""
<section class="panel">
  <div class="panel-title">PMPP Notes &middot; Programming Massively Parallel Processors, 5th ed.</div>
  <div class="md"><p>Chapter-by-chapter notes taken while reading the book. Cherry-picked
  highlights, worked examples and the parts most relevant to writing real CUDA kernels.</p></div>
  {chapter_list_html(chapters)}
</section>
"""
    (OUT / "pmpp" / "index.html").write_text(
        render_page(
            title="PMPP Notes · 100 Days of CUDA",
            subtitle="programming massively parallel processors &middot; chapter notes",
            content=pmpp_index,
            root="../",
            active="pmpp",
        ),
        encoding="utf-8",
    )

    # ---- pmpp/chapter-N.html ----
    for i, c in enumerate(chapters):
        body = md_to_html(c["text"])
        pager = ['<div class="pager">']
        if i > 0:
            p = chapters[i - 1]
            pager.append(f'<a href="chapter-{p["num"]}.html">&lsaquo; Ch{p["num"]:02d}: {p["title"]}</a>')
        pager.append('<span class="spacer"></span>')
        if i < len(chapters) - 1:
            n = chapters[i + 1]
            pager.append(f'<a href="chapter-{n["num"]}.html">Ch{n["num"]:02d}: {n["title"]} &rsaquo;</a>')
        pager.append("</div>")
        content = f"""
<article class="panel">
  <div class="panel-title">PMPP &middot; Chapter {c["num"]:02d} &middot; {c["title"]}</div>
  <div class="md">
{body}
  </div>
</article>
{''.join(pager)}
"""
        (OUT / "pmpp" / f'chapter-{c["num"]}.html').write_text(
            render_page(
                title=f'PMPP Ch{c["num"]}: {c["title"]} · 100 Days of CUDA',
                subtitle=f'pmpp chapter {c["num"]} &middot; {c["title"]}',
                content=content,
                root="../",
                active="pmpp",
                math=True,
            ),
            encoding="utf-8",
        )

    # ---- animations/index.html ----
    anim_index = f"""
<section class="panel">
  <div class="panel-title">Animations &middot; Interactive CUDA visualizations</div>
  <div class="md"><p>Step-by-step interactive animations of CUDA kernels. Run them, single-step
  them, and watch data move between global and shared memory.</p></div>
  {animation_list_html(anims)}
</section>
"""
    (OUT / "animations" / "index.html").write_text(
        render_page(
            title="Animations · 100 Days of CUDA",
            subtitle="interactive kernel visualizations",
            content=anim_index,
            root="../",
            active="anim",
        ),
        encoding="utf-8",
    )

    pages = len(chapters) + len(anims) + 3
    print(f"built {pages} pages -> {OUT}")


if __name__ == "__main__":
    build()
