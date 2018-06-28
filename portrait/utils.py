import numpy as np
from PIL import Image

# color map
LABEL_COLORS = [
  (0, 0, 0),  # 0=background
  # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
  (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
  # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
  (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
  # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
  (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
  # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
  (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]


def decode_labels(mask, num_images=1, num_classes=21):
  """Decode batch of segmentation masks.
  Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).
  Returns:
    A batch with num_images RGB images of the same size as the input.
  """
  n, h, w, _ = mask.shape
  assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
  outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
  for i in range(num_images):
    img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
    pixels = img.load()
    for j_, j in enumerate(mask[i, :, :, 0]):
      for k_, k in enumerate(j):
        if k < num_classes:
          pixels[k_, j_] = LABEL_COLORS[k]
    outputs[i] = np.array(img)
