# encode_known_faces_facenet.py
import os, pickle, numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

KNOWN_DIR = "faces/known"
embeddings = []
names = []

for person in os.listdir(KNOWN_DIR):
    pdir = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(pdir):
        continue
    print("Processing person:", person)
    person_images = [f for f in os.listdir(pdir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    for imgname in person_images:
        path = os.path.join(pdir, imgname)
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print("Cannot open", path, e); continue
        # detect and crop face
        face_tensor = mtcnn(img)
        if face_tensor is None:
            print("No face detected in", path)
            continue
        with torch.no_grad():
            emb = model(face_tensor.unsqueeze(0).to(device))
        emb_np = emb.cpu().numpy()[0]
        embeddings.append(emb_np)
        names.append(person)
        print("  -> encoded", imgname)
        
if len(embeddings) == 0:
    print("No embeddings created - add clearer images.")
else:
    data = {"embeddings": np.array(embeddings), "names": names}
    with open("faces/known_facenet.pkl", "wb") as f:
        pickle.dump(data, f)
    print("Saved encodings for", len(set(names)), "people. Total embeddings:", len(embeddings))
