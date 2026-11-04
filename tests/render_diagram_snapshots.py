"""Raster QA for this app's rect/line/polygon/text SVG subset, not a browser renderer.

Usage: python tests/render_diagram_snapshots.py SVG_DIRECTORY
Pillow and a readable TrueType font are required; no downloads or UI automation.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont


def render(path):
    root=ET.parse(path).getroot()
    _,_,width,height=map(float,root.attrib["viewBox"].split())
    scale=2
    image=Image.new("RGB",(int(width*scale),int(height*scale)),"#faf9f6")
    draw=ImageDraw.Draw(image)
    font_root=Path("/System/Library/Fonts/Supplemental")
    for element in root:
        kind=element.tag.rsplit("}",1)[-1];a=element.attrib
        number=lambda k:float(a[k])*scale
        if kind=="rect":
            xy=(number("x"),number("y"),number("x")+number("width"),number("y")+number("height"))
            draw.rounded_rectangle(xy,radius=float(a.get("rx",0))*scale,fill=a.get("fill"),outline=a.get("stroke"),width=scale)
        elif kind=="line":
            draw.line((number("x1"),number("y1"),number("x2"),number("y2")),fill=a.get("stroke","#26364c"),width=max(1,round(float(a.get("stroke-width",1))*scale)))
        elif kind=="polygon":
            points=[tuple(float(v)*scale for v in p.split(",")) for p in a["points"].split()]
            draw.polygon(points,fill=a.get("fill","#26364c"))
        elif kind=="text":
            name="Arial Bold.ttf" if a.get("font-weight")=="600" else "Arial.ttf"
            font_name=str(font_root/name) if (font_root/name).is_file() else "DejaVuSans.ttf"
            font=ImageFont.truetype(font_name,round(float(a.get("font-size",13))*scale))
            draw.text((number("x"),number("y")),element.text or "",fill=a.get("fill","#26364c"),font=font,anchor="ms" if a.get("text-anchor")=="middle" else "ls")
    image.save(path.with_suffix(".png"))


if __name__=="__main__":
    directory=Path(sys.argv[1])
    for path in sorted(directory.glob("*.svg")):render(path)
    print("Rendered static primitive snapshots; browser CSS/font/accessibility QA is separate.")
