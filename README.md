# DermAssist

## 📌 Overview
**DermAssist** is an AI-powered dermatology chatbot that leverages **LLaVA-Next** and **Mistral-7B** to provide comprehensive skin analysis and diagnostics. Designed for inclusivity, it supports a diverse range of skin conditions across different skin types. Users can submit images and descriptions of skin conditions to receive insights and recommendations, helping with early detection and management.

---

## ✨ Features
- **Multimodal AI Processing**: Uses **LLaVA-Next** for image analysis and **Mistral-7B** for natural language understanding.
- **Inclusive Dermatology Insights**: Supports a wide array of skin conditions across various skin types.
- **SCIN Dataset Integration**: Trained on Google’s open-source **SCIN dermatology dataset** for robust and accurate predictions.
- **Early Detection & Management**: Provides actionable insights to aid in identifying skin conditions early.
- **User-Friendly Interface**: Simplifies complex dermatological information into understandable insights.

---

## 🛠️ Installation & Setup
```sh
# Clone the repository
git clone https://github.com/yourusername/DermAssist.git
cd DermAssist

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

---

## 📖 How It Works
1. **Upload an Image**: Submit an image of the affected skin area.
2. **Describe the Condition**: Provide details about symptoms, duration, and any previous treatments.
3. **AI Analysis**: The model processes the image and text to generate insights.
4. **Receive Recommendations**: Get AI-driven insights and potential next steps for care.

---

## 🚀 Project Structure
```
DermAssist/
│-- models/            # Pre-trained AI models
│-- data/              # SCIN dataset and preprocessing scripts
│-- scripts/           # Utility scripts for training and evaluation
│-- app.py             # Main application script
│-- requirements.txt   # Python dependencies
│-- README.md          # Project documentation
```

---

## 🚧 Challenges
- **Generating clinically appropriate Q&A pairs** for teledermatology required expert knowledge.
- **Balancing computational efficiency and real-time processing** of AI models.
- **Ensuring inclusivity in diagnostics** across diverse skin tones and conditions.
- **Creating an intuitive and natural interaction** for users.

---

## 🏆 Achievements
- **Democratizing dermatology** by making AI-powered skin insights accessible to users globally.
- **Robust AI Model** trained on **SCIN dataset** for accurate and inclusive skin condition analysis.
- **User-Friendly UI** that simplifies complex dermatological data into actionable insights.

