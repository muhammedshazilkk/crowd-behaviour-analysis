# test_facenet.py
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os

print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=torch.device('cpu'))
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img_path = 'test_weapon.jpg'  # reuse any image you already have, or take a selfie and save as this name
if not os.path.exists(img_path):
    print("Place a test image named test_weapon.jpg in the project folder and re-run.")
    exit()

img = Image.open(img_path).convert('RGB')
face_tensor = mtcnn(img)
if face_tensor is None:
    print("MTCNN did not detect a face in the image.")
else:
    print("MTCNN detected a face tensor with shape:", face_tensor.shape)
    with torch.no_grad():
        emb = resnet(face_tensor.unsqueeze(0))
    print("Embedding shape:", emb.shape)
    print("Test OK.")
