🌿 GreenClassify
Deep Learning Web App for Multi-Vegetable Detection & Classification

GreenClassify is an end-to-end Deep Learning powered web application that detects, classifies and counts vegetables from an uploaded image in real-time using a trained CNN model deployed with Flask.

This project was developed as part of the SkillWallet / SmartInternz Deep Learning Internship Program.

🚀 NEW IMPROVEMENTS (Latest Version)

✔ Detects multiple vegetables in a single image
✔ Counts number of vegetables present
✔ Prevents wrong predictions for non-vegetable images
✔ Improved UI and prediction workflow
✔ Clean project structure & GitHub ready

🎯 Project Goal

Build a real-world AI web app that can:

• Automatically identify vegetables from images
• Avoid false predictions for unrelated objects
• Support multi-object detection & counting
• Deploy the trained model as a Flask web app
• Provide a simple and interactive user interface

🧠 Deep Learning Model
Feature	Details
Model	MobileNetV2
Technique	Transfer Learning
Framework	TensorFlow / Keras
Input Size	224 × 224 × 3
Pretrained On	ImageNet
Why MobileNetV2?

• Lightweight & fast inference
• Ideal for real-time web apps
• High accuracy with low compute cost
• Perfect for deployment environments

🏗️ Model Pipeline
Input Image
   ↓
Pre-processing (Resize + Normalize)
   ↓
MobileNetV2 Backbone
   ↓
Global Average Pooling
   ↓
Dense Softmax Layer
   ↓
Vegetable Predictions + Count
🧩 Key Features
🥕 Multi-Vegetable Detection

Detects multiple vegetables in one image and returns prediction list.

🔢 Vegetable Counting

Displays how many vegetables are detected.

🚫 Non-Vegetable Rejection

Model avoids random predictions if image is unrelated.

🌐 Web Application

User can upload image and get predictions instantly.

🎨 Modern UI

Responsive, animated and smooth interface.

🛠️ Tech Stack
Backend

• Python
• Flask
• TensorFlow / Keras
• NumPy & PIL

Frontend

• HTML5
• CSS3
• JavaScript

Tools

• Kaggle (Training Dataset)
• VS Code
• Git & GitHub

📂 Updated Project Structure
VEGETABLE_CLASSIFICATION/
│
├── static/
│   ├── css/
│   ├── uploads/
│   └── background.jpg
│
├── templates/
│   └── index.html
│
├── snapshots/              # App screenshots
├── app.py                  # Flask backend
├── vegetable_classifier_model.h5
├── requirements.txt
├── verify_fix.py
└── README.md

⚠ Dataset, virtual environment and cache files are excluded from GitHub.

🔄 Project Workflow
1️⃣ Data Collection

Vegetable dataset from Kaggle.

2️⃣ Data Pre-processing

• Resize images to 224×224
• Normalize pixel values
• Data augmentation

3️⃣ Model Training

• Transfer learning using MobileNetV2
• Adam optimizer
• Early stopping

4️⃣ Model Evaluation

Validation accuracy monitoring.

5️⃣ Deployment

Model integrated into Flask web app.

📊 Real-World Use Cases

🥕 Smart vegetable sorting machines
🛒 Retail inventory automation
🌾 Agri-tech crop identification
📦 Food supply chain automation

🖼️ Application Features

• Upload image
• Preview before prediction
• Multi-class vegetable prediction
• Vegetable counting
• Smooth UI animations

💻 Run Project Locally (Simple Steps)

Dataset Kaggle Link : https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset

1️⃣ Clone Repository
git clone https://github.com/Shree-2516/GreenClassify.git
cd GreenClassify
2️⃣ Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
python app.py
5️⃣ Open Browser
http://127.0.0.1:5000

Upload an image and test the model 🎉

📦 Deliverables

✔ Trained Deep Learning Model
✔ Flask Web Application
✔ Source Code
✔ Documentation
✔ Demo UI

👨‍💻 Author

Shreeyash Paraj
Deep Learning Intern – SkillWallet / SmartInternz

📜 License

This project is for educational & internship purposes only.


"if __name__ == "__main__":
    app.run(debug=True)
"