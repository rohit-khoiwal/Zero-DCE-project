import cv2
import jax
import jax.numpy as jnp
import enhance_image
from tqdm import tqdm


video_name = input("Enter video name with extension: ")
video_path = "tests/" + video_name

try:
    vid = cv2.VideoCapture(video_path)
except Exception as e:
    print("Video is not able to loaded. Please try again.")
    exit()

fps = vid.get(5)
out = cv2.VideoWriter(f"tests/enhanced_{video_name}", cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))



total_frames = vid.get(7)
print("Total : ",total_frames," frames.")

success, img = vid.read()
for i in tqdm(range(int(total_frames))):
    real_size = img.shape
    img = cv2.resize(img, (256,256))
    img = jnp.array(img)/255

    new_image = enhance_image.enhance_image(img, (640, 480), False)
    out.write(new_image)
    success, img = vid.read()

print("Enhancing video is completed.")