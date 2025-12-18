def open_image(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    return data

def save_image(img, output_path):
    with open(output_path, "wb") as f:
        f.write(img)
    print(f'save to {output_path}.')

def save_pil_image(pil_img, output_path):
    pil_img.save(output_path)
    print(f"save to {output_path} (size: {pil_img.size[0]}×{pil_img.size[1]})")