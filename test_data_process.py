import torch 
import numpy as np 
import PIL.Image as Image
import cv2

def process(
    image: np.ndarray,
    size: int = 512,
) -> torch.Tensor:
    image = cv2.resize(
        image,
        (
            size,
            size,
        ),
        interpolation=cv2.INTER_CUBIC,
    )
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0
    return torch.from_numpy(image).permute(
        2,
        0,
        1,
    )

if __name__ == "__main__":
    img = Image.open("/home/stevexu/VSprojects/ELITE/assets/images/girl.png")
    print(type(img))
    img = img.convert("RGB")
    print(type(img))
    img_np = np.array(img)
    processed_img = process(img_np)
    print(processed_img)