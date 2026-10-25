from PIL import Image, ImageDraw


def panel(path, destination):
    labels = ["source", "reference", "text", "reference-conditioned", "regional"]
    files = ["source", "reference", "text", "reference-edited", "regional"]
    canvas = Image.new("RGB", (5 * 160, 184), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, name) in enumerate(zip(labels, files)):
        with Image.open(path / (name + ".png")) as im:
            canvas.paste(im.convert("RGB").resize((160, 160)), (160 * i, 24))
        draw.text((160 * i + 3, 4), label, fill="black")
    canvas.save(destination)
