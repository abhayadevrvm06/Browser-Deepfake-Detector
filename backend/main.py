from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pillow_heif import register_heif_opener
from facenet_pytorch import MTCNN
import torch
import timm
from torchvision import transforms
import io

register_heif_opener()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(keep_all=False, device=device)

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABELS = ["fake", "real"]

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        return {
            "face_detected": False,
            "prediction": None,
            "confidence": None,
            "fake_probability": None,
            "real_probability": None,
        }

    box = boxes[0]
    face_crop = image.crop((box[0], box[1], box[2], box[3]))

    tensor = transform(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    prediction = LABELS[probs.argmax().item()]

    return {
        "face_detected": True,
        "prediction": prediction,
        "confidence": max(fake_prob, real_prob),
        "fake_probability": fake_prob,
        "real_probability": real_prob,
    }