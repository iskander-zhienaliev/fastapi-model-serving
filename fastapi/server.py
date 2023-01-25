import io

# from segmentation import get_segmentator, get_segments
from starlette.responses import Response
import torchvision.models as models
from torchvision.utils import save_image
import pickle
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, File, UploadFile

app = FastAPI(
    title="DeepLabV3 image segmentation",
    description="""Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 6)

    def forward(self, x):
        x = self.densenet(x)
        return x

model = MyModel()
model.to(device)
# model = models.vgg16(pretrained=True)
# model = torch.hub.load('pytorch/vision', 'vgg16', pretrained=True)
# inference_model_tensor = torch.load("./best_model.pth", map_location=torch.device('cpu'))
model.load_state_dict(torch.load("./best_model.pth", map_location=torch.device('cpu')))
model.eval()


def transform(image: Image):
    transform_val = transforms.Compose([
        transforms.CenterCrop((381, 297)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform_val(image)

@app.post("/segmentation")
async def predict(file: bytes = File(...)):
    # with open(file.filename, "wb") as f:
    #     f.write(await file.read())
    image = Image.open(io.BytesIO(file)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    # print('kek', image)

    labels_list = ['4c', '5', '2', '4a', '4b', '3']
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_label = preds.cpu().numpy()[0]
    return Response(labels_list[predicted_label], media_type="text/plain")