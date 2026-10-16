# Plant Disease Predictor

A Flask and TensorFlow web application that predicts plant diseases from leaf
images. The application provides:

- A browser-based interface for uploading a leaf image.
- A `POST /predict` API that can be used by React, Android, mobile, or other
  applications.
- A trained TensorFlow/Keras model based on MobileNetV2 transfer learning.
- A training script for retraining the model with the dataset structure used by
  this project.

> **Important:** This is an image-classification project, not a replacement for
> an agricultural expert. A prediction should be treated as an indication, not
> as a guaranteed diagnosis or treatment recommendation.

---

## 1. Demo and project status

The application loads the trained model from:

```text
models/plant_disease_model.keras
```

and the class-number mapping from:

```text
models/class_indices.json
```

If both files are present, the app can make predictions immediately. The
training dataset is not required to run the already-trained application.

The application was tested locally with a validation image and returned a
prediction such as:

```json
{
  "disease": "Apple - Apple scab",
  "confidence": "99.40%"
}
```

The exact result depends on the uploaded image.

---

## 2. Features

- Upload JPG, JPEG, PNG, and other image formats supported by Pillow.
- Maximum upload size: **10 MB**.
- Automatic image validation.
- Automatic RGB conversion and resizing to `224 x 224`.
- Pixel normalization from `0-255` to `0-1`.
- Prediction confidence score.
- Low-confidence response when confidence is below `50%`.
- JSON error responses for missing, invalid, or oversized files.
- Usable as a standalone web application or as a prediction API.

---

## 3. Machine-learning details

The training code is in [train_model.py](./train_model.py).

| Setting | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Base model | MobileNetV2 |
| Image size | `224 x 224 x 3` |
| Transfer-learning weights | ImageNet |
| Base model layers | Frozen during training |
| Custom layers | GlobalAveragePooling2D → Dense(128, ReLU) → Dense(number of classes, softmax) |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Default epochs | 10 |
| Batch size | 32 |
| Training augmentation | Rotation, shifts, horizontal flip, zoom |
| Validation preprocessing | Rescaling only |
| Best-model selection | Highest validation accuracy |
| Early stopping | Monitors validation loss with patience 5 |

The application and training script use the same preprocessing:

```python
image = image.convert("RGB").resize((224, 224))
image = np.array(image) / 255.0
```

Keeping preprocessing the same during training and prediction is important.

---

## 4. Dataset

This project uses the **New Plant Diseases Dataset**, downloaded from Kaggle:

<https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>

The dataset is commonly distributed with `train` and `valid` directories and
plant-disease class folders. The project currently contains the 38-class
structure from that dataset, including classes for apple, blueberry, cherry,
corn, grape, orange, peach, pepper, potato, raspberry, soybean, squash,
strawberry, and tomato.

### Dataset attribution and license

Please read and follow the dataset's current Kaggle license and usage
conditions before redistributing the images or using them commercially. Kaggle
dataset terms can change, so the Kaggle page above is the authoritative source.
This repository should not be treated as a substitute for the dataset
provider's attribution or license requirements.

### Dataset directory structure

The training script expects this structure:

```text
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image-1.jpg
│   │   └── image-2.jpg
│   ├── Apple___healthy/
│   ├── Tomato___Early_blight/
│   └── ...
└── valid/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    ├── Tomato___Early_blight/
    └── ...
```

Every class folder in `valid` should have a corresponding class folder in
`train`. The folder names become the class labels.

---

## 5. Repository structure

```text
Plant-Diseas_Predictor/
├── app.py                       # Flask web app and prediction API
├── train_model.py               # Model training script
├── train_model.ipynb            # Optional notebook version of training work
├── requirements.txt             # Python dependencies
├── README.md
├── models/
│   ├── plant_disease_model.keras # Trained model
│   └── class_indices.json        # Class-name mapping
├── templates/
│   └── index.html               # Web interface
├── static/
│   └── style.css                # Web-interface styles
└── dataset/
    ├── train/                   # Training images
    └── valid/                   # Validation images
```

The dataset is ignored by Git in this project because it is large. Download it
again from Kaggle when retraining is required.

---

## 6. Requirements

Recommended:

- Python **3.10 or 3.11**
- `pip`
- At least 4 GB RAM for inference; more RAM and a GPU are recommended for
  training.
- Internet access for the first installation and for downloading ImageNet
  weights if they are not cached.

TensorFlow support varies by Python version and operating system. Python 3.11
is a practical choice for deployment. The included virtual environment may use
a different version, but a new environment should follow the compatibility
requirements of the TensorFlow release installed on the machine.

---

## 7. Local installation on Windows

Open PowerShell in the project directory:

```powershell
cd "C:\path\to\Plant-Diseas_Predictor"
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, either allow local scripts for your user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

or run the environment's Python executable directly without activation:

```powershell
.\venv\Scripts\python.exe --version
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the important packages:

```powershell
python -c "import flask, tensorflow, numpy, PIL; print('Dependencies are ready')"
```

---

## 8. Run the web application locally

From the project root, with the virtual environment active:

```powershell
python app.py
```

Open this URL in a browser:

<http://127.0.0.1:5000/>

The app listens on `127.0.0.1:5000` by default. Stop it with `Ctrl+C`.

### Optional Flask debug mode

Debug mode is disabled by default. It can be enabled for local development:

PowerShell:

```powershell
$env:FLASK_DEBUG = "true"
python app.py
```

Do not enable Flask debug mode on a public production server.

---

## 9. Use the web interface

1. Start the application with `python app.py`.
2. Open <http://127.0.0.1:5000/>.
3. Select a clear plant-leaf image.
4. Click **Analyze Image**.
5. Review the predicted disease and confidence score.

For better results:

- Use a clear, well-lit image.
- Keep the leaf visible and reasonably centered.
- Avoid very blurry, dark, or heavily obstructed images.
- Use images that are similar to the classes represented in the training data.

---

## 10. Prediction API

### Endpoint

```text
POST /predict
```

The endpoint accepts `multipart/form-data` with a file field named `file`.

### cURL example

Local server:

```powershell
curl.exe -X POST `
  -F "file=@C:\path\to\leaf.jpg" `
  http://127.0.0.1:5000/predict
```

Linux/macOS:

```bash
curl -X POST \
  -F "file=@/path/to/leaf.jpg" \
  http://127.0.0.1:5000/predict
```

### Successful response

```json
{
  "disease": "Apple - Apple scab",
  "confidence": "99.40%"
}
```

### Low-confidence response

If the highest prediction confidence is below `50%`, the API returns:

```json
{
  "disease": "Not Recognized or Low Confidence",
  "confidence": "42.10%",
  "message": "The model is not confident enough to classify this correctly."
}
```

### Error responses

No file:

```json
{
  "error": "No file uploaded"
}
```

No selected file:

```json
{
  "error": "No selected file"
}
```

Model unavailable:

```json
{
  "error": "Prediction service is not ready. Train the model and ensure class_indices.json exists."
}
```

Invalid image:

```json
{
  "error": "The uploaded file is not a valid image."
}
```

Image larger than 10 MB:

```json
{
  "error": "Image is too large. Maximum size is 10 MB."
}
```

### HTTP status codes

| Status | Meaning |
|---:|---|
| 200 | Prediction completed |
| 400 | Missing, empty, or invalid image upload |
| 413 | Image is larger than 10 MB |
| 500 | Unexpected prediction or model-output error |
| 503 | Model or class mapping is not ready |

---

## 11. Call the API from another Python project

The simplest integration is to deploy this application once and call its
`/predict` endpoint from another project.

Install the HTTP client:

```bash
pip install requests
```

Example:

```python
import requests


API_URL = "https://your-deployed-app.example.com/predict"

with open("leaf.jpg", "rb") as image_file:
    response = requests.post(
        API_URL,
        files={"file": ("leaf.jpg", image_file, "image/jpeg")},
        timeout=60,
    )

response.raise_for_status()
result = response.json()

print("Disease:", result["disease"])
print("Confidence:", result["confidence"])
```

For a local API, use:

```python
API_URL = "http://127.0.0.1:5000/predict"
```

If the API returns an error JSON, inspect it before reading prediction fields:

```python
response = requests.post(
    API_URL,
    files={"file": open("leaf.jpg", "rb")},
    timeout=60,
)

data = response.json()

if response.ok:
    print(data["disease"], data["confidence"])
else:
    print("Prediction failed:", data.get("error", "Unknown error"))
```

In production code, use a context manager when opening files:

```python
with open("leaf.jpg", "rb") as image_file:
    response = requests.post(
        API_URL,
        files={"file": image_file},
        timeout=60,
    )
```

---

## 12. Call the API from JavaScript or React

```javascript
async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch(
    "https://your-deployed-app.example.com/predict",
    {
      method: "POST",
      body: formData
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || "Prediction failed");
  }

  return data;
}
```

Example browser input:

```javascript
document
  .querySelector("#imageInput")
  .addEventListener("change", async (event) => {
    const imageFile = event.target.files[0];

    if (!imageFile) {
      return;
    }

    try {
      const result = await predictDisease(imageFile);
      console.log(result.disease);
      console.log(result.confidence);
    } catch (error) {
      console.error(error.message);
    }
  });
```

Example HTML:

```html
<input id="imageInput" type="file" accept="image/*">
```

### CORS

If the frontend and Flask API are hosted on different domains, browsers may
block the request because of CORS. For a separate frontend, install:

```bash
pip install flask-cors
```

Then add the dependency to `requirements.txt` and configure [app.py](./app.py):

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/predict": {
            "origins": ["https://your-frontend.example.com"]
        }
    },
)
```

Do not use unrestricted `CORS(app)` in a production application unless the API
is intentionally public.

---

## 13. Use the model directly in another Python project

You do not need to call the Flask API if the other project is also Python and
can install TensorFlow. Copy these two files into the other project:

```text
models/
├── plant_disease_model.keras
└── class_indices.json
```

Install:

```bash
pip install tensorflow Pillow numpy
```

Create a prediction module:

```python
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model.keras"
CLASS_INDICES_PATH = BASE_DIR / "models" / "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with CLASS_INDICES_PATH.open(encoding="utf-8") as file:
    indices = json.load(file)

class_names = {
    int(index): name
    for name, index in indices.items()
}


def predict_disease(image_path):
    with Image.open(image_path) as source:
        image = source.convert("RGB").resize((224, 224))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)
    class_index = int(np.argmax(predictions, axis=1)[0])
    confidence = float(np.max(predictions))
    disease = class_names.get(class_index)

    if disease is None:
        raise RuntimeError("Model output does not match class_indices.json")

    return {
        "disease": disease.replace("___", " - ").replace("_", " "),
        "confidence": confidence,
    }


result = predict_disease("leaf.jpg")
print(result["disease"])
print(f"{result['confidence'] * 100:.2f}%")
```

Use the API approach when many different applications need to share one model.
Use direct loading when the Python application needs local inference and can
handle the TensorFlow dependency and model file.

---

## 14. Retrain the model

Retraining requires the dataset. The existing trained model is not enough to
run `train_model.py`; the `dataset/train` and `dataset/valid` folders must be
present.

1. Download the dataset from Kaggle:
   <https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>
2. Extract it into the project.
3. Confirm that `dataset/train` and `dataset/valid` exist.
4. Confirm that both directories contain class subdirectories.
5. Run:

```powershell
python train_model.py
```

The script will:

1. Read the class folders using `flow_from_directory`.
2. Resize images to `224 x 224`.
3. Apply augmentation to training images.
4. Load MobileNetV2 with ImageNet weights.
5. Add the project-specific classification layers.
6. Train for up to 10 epochs by default.
7. Save the best model to `models/plant_disease_model.keras`.
8. Save the class mapping to `models/class_indices.json`.

### Change the number of epochs

PowerShell:

```powershell
$env:EPOCHS = "20"
python train_model.py
```

Command Prompt:

```cmd
set EPOCHS=20
python train_model.py
```

The training script uses early stopping, so training may finish before the
configured maximum number of epochs.

### Training warnings

- The first training run may download MobileNetV2 ImageNet weights.
- Training the complete dataset can take a long time on CPU.
- Do not delete `class_indices.json`; the application needs it to translate a
  model output number into a disease name.
- If you change the class folders, retrain the model and regenerate the class
  mapping together.

---

## 15. Deploy online

The application can be deployed to a Python-capable hosting provider such as
Render, Railway, Fly.io, or a cloud VM. The exact dashboard names can change,
but the required concepts are the same:

- Install dependencies from `requirements.txt`.
- Start a production WSGI server.
- Keep the `models/` directory in the deployed project.
- Do not deploy the large training dataset unless retraining on the server.

### Render deployment

#### Step 1: Prepare the repository

Push these files and directories to GitHub:

```text
app.py
requirements.txt
models/
templates/
static/
```

The `dataset/` directory is not required for prediction. Keeping it out of the
deployment reduces repository and build size.

Add a production server to `requirements.txt`:

```text
gunicorn>=21.2.0
```

If TensorFlow or your hosting provider requires a specific Python version, add
the provider's supported Python-version configuration. Python 3.11 is a good
starting point for TensorFlow deployments.

#### Step 2: Create the Render service

1. Create an account at <https://render.com/>.
2. Choose **New → Web Service**.
3. Connect the GitHub repository.
4. Select the project root as the service root.
5. Use these commands:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

6. Choose a plan with enough memory for TensorFlow and the model.
7. Deploy the service.

After deployment, Render gives you a URL similar to:

```text
https://plant-disease-predictor.onrender.com
```

Open the homepage at:

```text
https://plant-disease-predictor.onrender.com/
```

Call the API at:

```text
https://plant-disease-predictor.onrender.com/predict
```

Example:

```bash
curl -X POST \
  -F "file=@leaf.jpg" \
  https://plant-disease-predictor.onrender.com/predict
```

### Deployment checklist

- [ ] `models/plant_disease_model.keras` is present.
- [ ] `models/class_indices.json` is present.
- [ ] `templates/index.html` is present.
- [ ] `static/style.css` is present.
- [ ] `gunicorn` is in `requirements.txt`.
- [ ] The start command is `gunicorn app:app`.
- [ ] The service has enough RAM for TensorFlow.
- [ ] Debug mode is disabled.
- [ ] CORS is restricted if a separate frontend is used.
- [ ] The public endpoint has been tested with a real image.

### Hosting considerations

- Free hosting plans may put inactive services to sleep, causing a slow first
  request.
- TensorFlow model loading can take time during service startup.
- The model file must be included in the deployment artifact or downloaded
  securely during deployment.
- Public APIs should eventually use authentication, rate limiting, logging,
  and monitoring.
- Do not commit API keys, passwords, or private configuration files.

---

## 16. Troubleshooting

### `ModuleNotFoundError`

Activate the correct virtual environment and install dependencies:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### TensorFlow installation fails

Check the Python and operating-system compatibility for the TensorFlow version
you are installing. Python 3.10 or 3.11 is generally a safer choice than
using a newly released Python version.

### `Prediction service is not ready`

Confirm that both files exist:

```text
models/plant_disease_model.keras
models/class_indices.json
```

Also check the application logs for a model-loading error.

### `The uploaded file is not a valid image`

Use a real JPG, JPEG, or PNG file. Renaming a non-image file to `.jpg` does not
make it a valid image.

### `Image is too large`

The current application accepts images up to 10 MB. Resize or compress the
image before uploading, or change `MAX_CONTENT_LENGTH` in [app.py](./app.py)
deliberately after considering server limits.

### API works locally but not from React

Check:

1. The frontend is using the correct deployed URL.
2. The request uses `POST`.
3. The form field name is exactly `file`.
4. The frontend origin is allowed by the Flask CORS configuration.
5. The browser's Network and Console tabs for the actual response.

### Model gives poor predictions

Try a clear image similar to the training images. Also verify that:

- The input is a plant leaf.
- The class is represented in the training dataset.
- Training and inference preprocessing match.
- The correct `class_indices.json` belongs to the loaded model.
- The model was not replaced without replacing its class mapping.

---

## 17. Security and production notes

The included Flask server is suitable for development and small demonstrations.
For production:

- Use Gunicorn or another production WSGI server instead of Flask's
  development server.
- Keep debug mode disabled.
- Validate uploads and keep the 10 MB limit or set an intentional limit.
- Restrict CORS to trusted frontend origins.
- Add authentication if the endpoint is private.
- Add rate limiting to prevent abuse.
- Store logs without exposing secrets or unnecessary personal data.
- Keep dependencies updated and review TensorFlow security advisories.
- Do not return internal exception details to users.

---

## 18. License and attribution

Check the repository's license before redistributing this code. The dataset
has separate terms controlled by its Kaggle source:

<https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset>

Always follow the dataset provider's attribution, license, and commercial-use
requirements.
