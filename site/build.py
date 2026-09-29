"""Assemble thinkreasonlearn.com from the site pages and the Sphinx docs.

    python site/build.py <sphinx html dir> <output dir> [--local]

Layout: /index.html, /founderbrain/, /privacy/, /research/, /fonts/, /img/ and /docs/.
A plain static server (such as the docs container's python -m http.server) then
serves /founderbrain, /privacy and /research as clean addresses. The docs used to
live at the site root, so every old docs address (such as /modules.html) gets a
small page that forwards to its new place under /docs. --local rewrites the site's
own absolute links to relative ones, so a local preview never leaves your machine.
"""

import html as htmllib
import pathlib
import re
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
PAGES = {
    "index.html": "index.html",
    "founderbrain.html": "founderbrain/index.html",
    "privacy.html": "privacy/index.html",
    "research.html": "research/index.html",
}
# old docs pages whose content now lives on a site page
MOVED = {"research.html": "/research"}
FORWARD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved</title>
<link rel="canonical" href="https://thinkreasonlearn.com{0}">
<script>location.replace("{0}" + location.hash)</script>
<meta http-equiv="refresh" content="0; url={0}">
</head>
<body><p>This page has moved to <a href="{0}">{1}</a>.</p></body>
</html>
"""


def local_links(text):
    """Point the site's own links at the local preview, keeping canonical links."""
    return re.sub(
        r'(?<!rel="canonical" )href="https://thinkreasonlearn\.com/', 'href="/', text
    )


def main(docs, out, local):
    """Build the site from the Sphinx output in docs into out."""
    docs, out = pathlib.Path(docs), pathlib.Path(out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(docs, out / "docs")
    shutil.copytree(HERE / "fonts", out / "fonts")
    shutil.copytree(HERE / "img", out / "img")
    for src, dst in PAGES.items():
        text = (HERE / src).read_text()
        text = text.replace('href="fonts/fonts.css"', 'href="/fonts/fonts.css"')
        if not text.lstrip().lower().startswith("<!doctype"):
            text = (
                '<!doctype html>\n<html lang="en">\n<head>\n'
                + text.replace("</style>", "</style>\n</head>\n<body>", 1)
                + "\n</body>\n</html>\n"
            )
        if local:
            text = local_links(text)
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for page in (out / "docs").rglob("*.html"):
        rel = page.relative_to(out / "docs").as_posix()
        if local:
            page.write_text(
                page.read_text().replace(
                    'href="https://thinkreasonlearn.com/', 'href="/'
                )
            )
        old = out / rel
        if old.exists() or rel.startswith("_"):
            continue
        new = MOVED.get(rel, "/docs/" + re.sub(r"(^|/)index\.html$", r"\1", rel))
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text(FORWARD.format(new, htmllib.escape(new)))
    print(f"built {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--local" in sys.argv)
