"""Générateur de logo complet MEMORIX (icône + typographie dégradée)."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip("#")
    return tuple(int(hex_code[i : i + 2], 16) for i in (0, 2, 4))


def interpolate_color(c1, c2, factor):
    return tuple(int(a + (b - a) * factor) for a, b in zip(c1, c2))


def draw_linear_gradient_text(draw, text, pos, font, c_start, c_end):
    """Dessine un texte avec un dégradé horizontal progressif."""
    bbox = draw.textbbox(pos, text, font=font)
    t_w = bbox[2] - bbox[0]
    t_h = bbox[3] - bbox[1]

    # Crée un masque du texte
    mask = Image.new("L", (t_w + 20, t_h + 20), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.text((0, 0), text, font=font, fill=255)

    # Dégradé horizontal
    grad = Image.new("RGBA", (t_w + 20, t_h + 20))
    for x in range(t_w + 20):
        factor = min(1.0, max(0.0, x / max(1, t_w)))
        col = (*interpolate_color(c_start, c_end, factor), 255)
        for y in range(t_h + 20):
            grad.putpixel((x, y), col)

    # Applique le masque
    grad.putalpha(mask)
    return grad, pos


def make_memorix_badge(out_path="assets/logo.png", width=900, height=260):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    violet = hex_to_rgb("#7C5CFC")
    blue = hex_to_rgb("#4F8CFF")
    text_color = hex_to_rgb("#E8E8F5")
    sub_color = hex_to_rgb("#8C8CA8")

    # ---------------- 1. Icône Stylisée (gauche) ----------------
    icon_box_size = 180
    scale = icon_box_size / 48.0
    ox, oy = 40, 40

    fill_color = (*violet, 38)
    page_l = [(6, 38), (6, 12), (10, 8), (22, 8), (24, 10), (24, 38), (22, 36), (10, 36)]
    page_r = [(42, 38), (42, 12), (38, 8), (26, 8), (24, 10), (24, 38), (26, 36), (38, 36)]

    poly_l = [(ox + x * scale, oy + y * scale) for x, y in page_l]
    poly_r = [(ox + x * scale, oy + y * scale) for x, y in page_r]

    draw.polygon(poly_l, fill=fill_color)
    draw.polygon(poly_r, fill=fill_color)

    line_w = max(3, int(2.4 * scale))
    draw.line(poly_l + [poly_l[0]], fill=(*violet, 230), width=line_w)
    draw.line(poly_r + [poly_r[0]], fill=(*blue, 230), width=line_w)

    nodes = {
        "top": (ox + 24 * scale, oy + 14 * scale),
        "left": (ox + 16 * scale, oy + 22 * scale),
        "right": (ox + 32 * scale, oy + 22 * scale),
        "center": (ox + 24 * scale, oy + 28 * scale),
    }

    edges = [("top", "left"), ("top", "right"), ("left", "center"), ("right", "center")]
    for s_k, e_k in edges:
        draw.line([nodes[s_k], nodes[e_k]], fill=(*interpolate_color(violet, blue, 0.5), 210), width=max(2, int(1.8 * scale)))

    for name, r, col in [("top", 3.2 * scale, blue), ("left", 2.6 * scale, violet), ("right", 2.6 * scale, violet), ("center", 3.4 * scale, text_color)]:
        x, y = nodes[name]
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(*col, 255))

    # ---------------- 2. Typographie Stylisée ----------------
    # Chargement d'une police système ou par défaut
    font_main = None
    font_sub = None
    for font_name in ["DejaVuSans-Bold.ttf", "Arial-Bold.ttf", "arialbd.ttf", "Helvetica-Bold.ttf"]:
        try:
            font_main = ImageFont.truetype(font_name, 76)
            font_sub = ImageFont.truetype(font_name.replace("Bold", "").replace("bd", ""), 22)
            break
        except OSError:
            continue

    if font_main is None:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Titre "MEMORIX" en dégradé
    title_x = 245
    title_y = 62

    # Lueur diffuse derrière le texte
    glow_color = (*violet, 40)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((title_x + dx, title_y + dy), "MEMORIX", font=font_main, fill=glow_color)

    # Rendu du titre
    t_img, t_pos = draw_linear_gradient_text(draw, "MEMORIX", (title_x, title_y), font_main, violet, blue)
    img.paste(t_img, t_pos, t_img)

    # Sous-titre moderne
    sub_text = "ASSISTANT D'ÉTUDES & RAG INTELLIGENT"
    draw.text((title_x + 4, title_y + 88), sub_text, font=font_sub, fill=(*sub_color, 240))

    img.save(out_path, "PNG")
    print(f"Logo enregistré avec succès dans : {out_path}")


if __name__ == "__main__":
    make_memorix_badge()