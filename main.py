import torch
import os
import shutil
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


model = fasterrcnn_resnet50_fpn(pretrained=True)

output_folder = 'detected_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


image_files = ['semena.jpg']


transform = T.Compose([T.ToTensor()])

for image_file in image_files:

    image = Image.open(image_file)
    resized_image = image.resize((416, 416))

    input_image = transform(image)

model.eval()
with torch.no_grad():
    predictions = model([input_image])

result_image = image.copy()
draw = ImageDraw.Draw(result_image)

threshold  = 0.1
count = 0
for score, label, box in zip(predictions[0]['scores'], predictions[0]['labels'], predictions[0]['boxes']):
    if score > threshold:
        box = [int(b) for b in box]
        draw.rectangle(box, outline='red', width=10)
        draw.text((box[0], box[1]), f'Class: {label.item()}, Score: {score.item()}',fill = 'red')
        count +=1
        cropped_image = image.crop((box[0], box[1], box[3]))
        cropped_image.save(os.path.join(output_folder, f'detected_image_{label.item()}.jpg'))

plt.imshow(result_image)
plt.show()

print(f'Количество найденных зернышек на изображении: {count}')
