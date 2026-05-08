"""Transform a bd-generated DOT file to a dark-mode palette."""
import sys, re, pathlib

# Light → dark mapping. bd's palette uses `color=` for fills (per inspection).
PALETTE = {
    # node fills (light-mode → dark-mode)
    '#e8f4fd': '#1f3a5f',   # open  : light blue → deep blue
    '#d4edda': '#1f4d2e',   # closed (if used): light green → deep green
    '#fff3cd': '#5a4216',   # in_progress (if used): light amber → dark amber
    '#f8d7da': '#5a1d2c',   # blocked (if used): light red → dark red
    '#e2e3e5': '#2d2d2d',   # deferred (if used): light grey → dark grey
    # text
    '#1a1a1a': '#e6e6e6',   # near-black → near-white
    # edges
    '#666666': '#888888',   # medium grey edges → slightly brighter
}

def transform(text):
    for light, dark in PALETTE.items():
        text = text.replace(light, dark)
    # Inject background color and global font color after `digraph beads {`
    text = re.sub(
        r'(digraph beads \{)',
        r'\1\n  bgcolor="#0d1117";\n  graph [bgcolor="#0d1117"];\n  node [fontcolor="#e6e6e6"];\n  edge [color="#888888", fontcolor="#cccccc"];',
        text,
        count=1,
    )
    return text

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dst.write_text(transform(src.read_text()))
print(f"Wrote {dst} ({dst.stat().st_size} bytes)")
