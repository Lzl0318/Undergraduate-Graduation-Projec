from PIL import Image
import torchvision.transforms as transforms

img = Image.open('./NEU-CLS-64/pa/1.jpg')
transform1 = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
img_new = transform1(img)
