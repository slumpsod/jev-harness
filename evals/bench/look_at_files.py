"""The files evals/bench/look_at.py asks about, made from scratch each run (nothing to download, nothing personal): scans
and photos whose text is only in the pixels, and lookalikes whose text can be read as text. Every value the bench checks
an answer against is defined here.

    python evals/bench/look_at_files.py OUT_DIR     # write them, to look at by eye
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FORM = {"name": "Marisol Okafor", "dob": "03/14/1987", "policy": "HX-44721-B", "group": "88130", "phone": "(503) 555-0147"}
INVOICE = {"number": "INV-20931", "date": "September 30, 2026", "total": "$1,284.50", "vendor": "Larkspur Plumbing Supply"}
RECEIPT = {"store": "CORNER CROW BAKERY", "total": "48.37", "items": [("Sourdough loaf", "7.50"), ("Almond croissant x4", "17.60"),
           ("Oat milk latte", "5.25"), ("Birthday cake (8in)", "14.00")], "tax": "4.02", "card": "4417"}
SCREEN = {"title": "Rendering a whole city in Blender - 4 hour timelapse", "app": "Blender", "channel": "Polyfjord Studio"}
CSV_ROWS = [("2026-10-01", "Groceries", "86.12"), ("2026-10-03", "Fuel", "41.90"), ("2026-10-06", "Groceries", "52.33"),
            ("2026-10-09", "Internet", "65.00"), ("2026-10-12", "Pharmacy", "18.47")]   # total 263.82
NOTE = {"when": "Thu 3:45", "who": "Dr. Abernathy", "phone": "555-0182"}
LEASE = {"end": "July 31, 2027", "rent": "$1,950", "unit": "4B"}
REF = {"ref": "CLM-7730-QX", "amount": "$612.40"}
MEETING = {"owner": "Priya", "date": "November 4"}


def font(size: int, kind: str = "sans") -> ImageFont.ImageFont:
    names = {"sans": ["arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"], "mono": ["consola.ttf", "cour.ttf", "DejaVuSansMono.ttf"],
             "hand": ["segoepr.ttf", "Inkfree.ttf", "comic.ttf", "DejaVuSans-Oblique.ttf"]}[kind]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def scanned(img: Image.Image, seed: int, angle: float = 0.8) -> Image.Image:
    """A flatbed scan's look: slightly turned, grey speckles, soft."""
    rnd = random.Random(seed)
    img = img.convert("L").rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=235)
    px = img.load()
    w, h = img.size
    for _ in range(w * h // 300):
        x, y = rnd.randrange(w), rnd.randrange(h)
        px[x, y] = max(0, px[x, y] - rnd.randrange(40, 120))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def form_png(path: Path) -> None:
    img = Image.new("L", (1700, 2200), 255)
    d = ImageDraw.Draw(img)
    d.text((120, 110), "RIVERBEND FAMILY CLINIC", font=font(64), fill=0)
    d.text((120, 200), "New patient intake form", font=font(40), fill=60)
    rows = [("Full name", FORM["name"]), ("Date of birth", FORM["dob"]), ("Insurance policy no.", FORM["policy"]),
            ("Group no.", FORM["group"]), ("Phone", FORM["phone"]), ("Allergies", "Penicillin")]
    y = 360
    for label, value in rows:
        d.text((120, y), label + ":", font=font(38), fill=40)
        d.line((560, y + 48, 1550, y + 48), fill=120, width=2)
        d.text((580, y - 4), value, font=font(44, "hand"), fill=20)
        y += 150
    d.rectangle((120, y + 40, 1580, y + 400), outline=90, width=3)
    d.text((150, y + 70), "Signature", font=font(32), fill=80)
    d.line((300, y + 300, 900, y + 230), fill=30, width=5)
    scanned(img, 1).save(path)


def invoice_pages() -> list[Image.Image]:
    p1 = Image.new("L", (1700, 2200), 255)
    d = ImageDraw.Draw(p1)
    d.text((120, 120), INVOICE["vendor"].upper(), font=font(58), fill=0)
    d.text((120, 210), "1418 Industrial Way, Salem OR", font=font(32), fill=70)
    d.text((1150, 120), "INVOICE", font=font(70), fill=0)
    d.text((1150, 230), f"No. {INVOICE['number']}", font=font(36), fill=20)
    d.text((1150, 285), f"Date: {INVOICE['date']}", font=font(32), fill=20)
    d.text((120, 420), "Bill to: Harrow Street Apartments LLC", font=font(36), fill=20)
    y = 560
    for desc, qty, amt in [("PEX tubing 1/2in x 100ft", 4, "312.00"), ("Brass ball valve 3/4in", 12, "238.80"),
                           ("Water heater flex connector", 10, "96.50"), ("Labor - on-site delivery", 1, "85.00")]:
        d.text((120, y), desc, font=font(34), fill=10)
        d.text((1050, y), str(qty), font=font(34), fill=10)
        d.text((1350, y), amt, font=font(34), fill=10)
        y += 80
    d.text((120, 2050), "Continued on page 2", font=font(30), fill=90)
    p2 = Image.new("L", (1700, 2200), 255)
    d = ImageDraw.Draw(p2)
    d.text((120, 120), f"{INVOICE['number']} - page 2", font=font(32), fill=70)
    y = 260
    for desc, qty, amt in [("Copper elbow 3/4in (box)", 6, "214.20"), ("Thread sealant", 8, "41.60"), ("Shutoff key", 2, "18.90")]:
        d.text((120, y), desc, font=font(34), fill=10)
        d.text((1050, y), str(qty), font=font(34), fill=10)
        d.text((1350, y), amt, font=font(34), fill=10)
        y += 80
    d.text((1000, 700), "Subtotal: 1,007.00", font=font(36), fill=10)
    d.text((1000, 770), "Shipping: 185.00", font=font(36), fill=10)
    d.text((1000, 840), "Tax: 92.50", font=font(36), fill=10)
    d.text((1000, 940), f"TOTAL DUE: {INVOICE['total']}", font=font(46), fill=0)
    d.text((120, 1200), "Payment due within 30 days. Thank you for your business.", font=font(30), fill=60)
    return [scanned(p1, 2, -0.6), scanned(p2, 3, 0.5)]


def image_pdf(path: Path, pages: list[Image.Image]) -> None:
    """A scanner's PDF: each page only a picture, no text layer."""
    import pymupdf as fitz
    doc = fitz.open()
    for img in pages:
        tmp = path.with_suffix(f".p{len(doc)}.png")
        img.save(tmp)
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, filename=str(tmp))
        tmp.unlink()
    doc.save(path)


def text_pdf(path: Path, lines: list[str]) -> None:
    import pymupdf as fitz
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    for line in lines:
        page.insert_text((60, y), line, fontsize=12)
        y += 20
    doc.save(path)


def receipt_jpg(path: Path) -> None:
    """A phone photo of a till receipt on a wooden table, stored sideways with the EXIF tag that turns it upright."""
    rnd = random.Random(7)
    paper = Image.new("RGB", (620, 1250), (246, 243, 236))
    d = ImageDraw.Draw(paper)
    f, big = font(30, "mono"), font(40, "mono")
    d.text((90, 50), RECEIPT["store"], font=big, fill=(30, 30, 30))
    d.text((140, 110), "22 Mill St - Bend OR", font=f, fill=(40, 40, 40))
    d.text((150, 150), "10/14/2026  08:12", font=f, fill=(40, 40, 40))
    y = 240
    for item, amt in RECEIPT["items"]:
        d.text((40, y), item, font=f, fill=(25, 25, 25))
        d.text((480, y), amt, font=f, fill=(25, 25, 25))
        y += 52
    d.line((40, y + 10, 580, y + 10), fill=(60, 60, 60), width=2)
    d.text((40, y + 40), "TAX", font=f, fill=(25, 25, 25))
    d.text((480, y + 40), RECEIPT["tax"], font=f, fill=(25, 25, 25))
    d.text((40, y + 100), "TOTAL", font=big, fill=(10, 10, 10))
    d.text((450, y + 100), RECEIPT["total"], font=big, fill=(10, 10, 10))
    d.text((40, y + 180), f"VISA ****{RECEIPT['card']}", font=f, fill=(40, 40, 40))
    d.text((120, y + 260), "THANK YOU - COME AGAIN", font=f, fill=(40, 40, 40))
    paper = paper.rotate(-4, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))
    table = Image.new("RGB", (1500, 2000), (120, 84, 52))
    td = ImageDraw.Draw(table)
    for y in range(0, 2000, 9):   # wood grain
        td.line((0, y, 1500, y + rnd.randint(-6, 6)), fill=(110 + rnd.randint(-12, 12), 76 + rnd.randint(-8, 8), 46), width=3)
    mask = paper.convert("L").point(lambda v: 255 if v > 0 else 0)
    table.paste(paper, (420, 330), mask)
    shade = Image.new("L", table.size, 0)
    ImageDraw.Draw(shade).ellipse((-600, -400, 2100, 2600), fill=60)
    table = Image.composite(table, Image.new("RGB", table.size, (40, 28, 20)), shade.point(lambda v: 255 - v // 3))
    table = table.filter(ImageFilter.GaussianBlur(1.1))
    sideways = table.rotate(90, expand=True)   # the sensor's orientation; the tag says to turn it back
    exif = sideways.getexif()
    exif[0x0112] = 6
    sideways.save(path, quality=82, exif=exif)


def screenshot_png(path: Path) -> None:
    """A video page in a browser, the video showing a 3D program (the real case: session 20261012-153059-7780, "what
    software are they using in this video?", where the worker had no way to look at the screenshot it took)."""
    img = Image.new("RGB", (1920, 1080), (15, 15, 15))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 1920, 70), fill=(33, 33, 33))
    d.text((30, 18), "youtube.com/watch?v=Qm2xY81", font=font(26), fill=(200, 200, 200))
    d.rectangle((60, 100, 1340, 820), fill=(48, 48, 48))   # the player, showing the 3D program
    d.rectangle((60, 100, 1340, 140), fill=(70, 70, 70))
    d.text((80, 106), f"{SCREEN['app']} 4.5   File  Edit  Render  Window  Help", font=font(22), fill=(230, 230, 230))
    for i in range(14):   # a city in the viewport
        x = 200 + i * 75
        hgt = 120 + (i * 53) % 260
        d.rectangle((x, 700 - hgt, x + 55, 700), fill=(90 + i * 5, 110, 140), outline=(200, 160, 60))
    d.rectangle((60, 790, 1340, 820), fill=(25, 25, 25))
    d.rectangle((60, 800, 560, 808), fill=(230, 30, 30))
    d.text((60, 850), SCREEN["title"], font=font(40), fill=(245, 245, 245))
    d.text((60, 910), f"{SCREEN['channel']}  -  1.2M views", font=font(28), fill=(170, 170, 170))
    for i in range(6):
        d.rectangle((1400, 100 + i * 150, 1640, 235 + i * 150), fill=(60 + i * 10, 60, 70))
        d.text((1660, 120 + i * 150), f"Up next {i + 1}", font=font(24), fill=(200, 200, 200))
    img.save(path)


def note_jpg(path: Path) -> None:
    img = Image.new("RGB", (1400, 1400), (205, 200, 190))
    note = Image.new("RGB", (800, 800), (252, 236, 120))
    d = ImageDraw.Draw(note)
    hand = font(64, "hand")
    d.text((60, 90), "Dentist!!", font=font(80, "hand"), fill=(30, 40, 120))
    d.text((60, 260), f"{NOTE['when']}pm", font=hand, fill=(30, 40, 120))
    d.text((60, 380), NOTE["who"], font=hand, fill=(30, 40, 120))
    d.text((60, 500), f"call {NOTE['phone']}", font=hand, fill=(30, 40, 120))
    d.text((60, 620), "if running late", font=font(48, "hand"), fill=(30, 40, 120))
    note = note.rotate(7, resample=Image.BICUBIC, expand=True, fillcolor=(205, 200, 190))
    img.paste(note, (280, 260))
    img.filter(ImageFilter.GaussianBlur(0.9)).save(path, quality=80)


def lease_pages() -> list[Image.Image]:
    pages = []
    texts = [["RESIDENTIAL LEASE AGREEMENT", "", "This lease is made between Halvorsen Properties", "(\"Landlord\") and the Tenant named below.",
              f"Premises: 2210 Alder Ave, Unit {LEASE['unit']}"],
             ["1. TERM", "The term begins August 1, 2026.", "", "2. USE", "The premises are for residential use only.",
              "No business may be run from them."],
             ["3. RENT AND TERM END", f"Monthly rent: {LEASE['rent']}, due on the 1st.", f"This lease ends on {LEASE['end']}.",
              "A late fee of $75 applies after the 5th."]]
    for i, lines in enumerate(texts):
        img = Image.new("L", (1700, 2200), 255)
        d = ImageDraw.Draw(img)
        y = 180
        for line in lines:
            d.text((140, y), line, font=font(46 if y == 180 and i == 0 else 38), fill=15)
            y += 80
        d.text((780, 2080), f"Page {i + 1} of 3", font=font(28), fill=90)
        pages.append(scanned(img, 10 + i, [0.4, -0.7, 0.9][i]))
    return pages


def make(dest: Path) -> dict[str, str]:
    """Write every file into `dest` (made the same each time); -> {case file key: path}."""
    dest.mkdir(parents=True, exist_ok=True)
    out = {}

    def p(name):
        out[name] = str(dest / name)
        return dest / name
    form_png(p("intake_form_scan.png"))
    image_pdf(p("invoice_INV-scan.pdf"), invoice_pages())
    receipt_jpg(p("IMG_4471.jpg"))
    screenshot_png(p("screen.png"))
    text_pdf(p("invoice_INV-20931.pdf"), [
        INVOICE["vendor"], "1418 Industrial Way, Salem OR", "", f"INVOICE  No. {INVOICE['number']}", f"Date: {INVOICE['date']}",
        "Bill to: Harrow Street Apartments LLC", "", "PEX tubing 1/2in x 100ft   4   312.00", "Brass ball valve 3/4in   12   238.80",
        "Water heater flex connector   10   96.50", "Labor - on-site delivery   1   85.00", "Copper elbow 3/4in (box)   6   214.20",
        "Thread sealant   8   41.60", "Shutoff key   2   18.90", "", "Subtotal: 1,007.00", "Shipping: 185.00", "Tax: 92.50",
        f"TOTAL DUE: {INVOICE['total']}"])
    with open(p("october_expenses.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "category", "amount"])
        w.writerows(CSV_ROWS)
    note_jpg(p("PXL_20261016_sticky.jpg"))
    image_pdf(p("lease_signed_scan.pdf"), lease_pages())
    text_pdf(p("scan_0042.pdf"), ["Northgate Mutual - Claim acknowledgement", "", f"Claim reference: {REF['ref']}",
                                  f"Approved amount: {REF['amount']}", "Payment will be sent within 10 business days."])
    p("meeting_notes.txt").write_text(f"Kickoff - budget review\nDate: {MEETING['date']}\nOwner of the follow-up: {MEETING['owner']}\n"
                                      "Actions: send the revised budget to finance; book the venue.\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    for k, v in make(Path(sys.argv[1] if len(sys.argv) > 1 else "look_at_files")).items():
        print(k, v)
