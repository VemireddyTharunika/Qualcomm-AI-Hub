import torch
from torchvision import models, transforms
from PIL import Image
import requests
import qai_hub as hub

# Load pre-trained MobileNetV2 model
torch_model = models.mobilenet_v2(pretrained=True)
torch_model.eval()

# Trace model for on-device deployment
input_shape = (1, 3, 224, 224)
example_input = torch.rand(input_shape)
traced_torch_model = torch.jit.trace(torch_model, example_input)

# Choose a device
device = hub.Device("Samsung Galaxy S24")

# Optimize model for the chosen device
compile_job = hub.submit_compile_job(
    model=traced_torch_model,
    device=device,
    input_specs=dict(image=input_shape),
)

# Wait for compilation to complete
compiled_model = compile_job.get_target_model()

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
def load_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

image_path = "path_to_your_image.jpg"
image_tensor = load_image(image_path)

# Submit profile job to run the model on the hosted device
profile_job = hub.submit_profile_job(
    model=compiled_model,
    device=device,
    inputs=dict(image=image_tensor)
)

# Wait for the profiling job to complete and get the output
output = profile_job.get_outputs()['image']

# Get the predicted class
_, predicted_class = output.max(1)

# Load class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
response = requests.get(LABELS_URL)
labels = response.json()

# Get the label for the predicted class
predicted_label = labels[predicted_class.item()]

print(f"Predicted label: {predicted_label}")
