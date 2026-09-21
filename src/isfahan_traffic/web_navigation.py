from pathlib import Path


def add_map_navigation(path: Path) -> None:
    if not path.exists():
        return
    page = path.read_text(encoding="utf-8")
    if 'name="viewport"' not in page:
        page = page.replace("<head>", '<head><meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    header = '''<header id="traffic-map-navigation">
<a href="dashboard/dashboard.html#/intersections">Back to traffic analysis</a>
<h1>Intersection locations</h1>
<p>These are location candidates that need local checking &middot; background map tiles need internet access</p>
</header><style>
body{margin:0}
#traffic-map-navigation{font-family:Arial,sans-serif;padding:16px 24px;background:#f4f5ef;color:#283f35;border-bottom:1px solid #dce2d8;box-sizing:border-box}
#traffic-map-navigation a{color:#217568;font-size:14px}
#traffic-map-navigation a:focus-visible{outline:3px solid #217568;outline-offset:4px}
#traffic-map-navigation h1{font-size:21px;margin:8px 0}
#traffic-map-navigation p{font-size:13px;margin:0;line-height:1.5}
.plotly-graph-div{height:calc(100vh - 120px)!important;min-height:400px}
@media(max-width:600px){#traffic-map-navigation{padding:14px 18px}.plotly-graph-div{height:calc(100vh - 155px)!important}.plotly-graph-div .gtitle{display:none}}
</style>'''
    if '<header id="traffic-map-navigation">' in page:
        start = page.index('<header id="traffic-map-navigation">')
        end = page.index("</style>", start) + len("</style>")
        page = page[:start] + header + page[end:]
    else:
        page = page.replace("<body>", "<body>" + header, 1)
    path.write_text(page, encoding="utf-8")
