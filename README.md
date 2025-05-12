# AceBot - Academic AI Assistant

AceBot is an intelligent AI-powered academic assistant designed to help students understand course materials, manage their syllabus, and prepare for exams effectively. The application combines natural language processing with vector search to deliver contextually relevant academic information.

![alt text](https://github.com/Japjeet09/Academic_chatbot/blob/main/image.png?raw=true)

## 🌟 Features

### 💬 Smart Academic Chat
- Subject-specific conversations with AI
- Voice input capability for natural interaction
- Context-aware responses based on syllabus materials
- Persistent chat history for each subject

### 📚 Subject & Syllabus Management
- Create and organize subjects
- Upload syllabus via PDF or images
- Automatic topic extraction using OCR
- Manual syllabus editing and custom topic creation

### 📝 Exam Preparation Mode
- Upload and process previous year question papers
- Generate quiz questions with adaptive difficulty
- Track performance and identify weak areas
- Create probable exam questions based on pattern analysis

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **Database**: MySQL (user data, subjects, chat history)
- **Vector Database**: Qdrant (for embeddings and similarity search)
- **LLM**: Groq API (LLama 3.3 70B Versatile model)
- **OCR**: EasyOCR for text extraction from images
- **Speech Recognition**: SpeechRecognition library for voice input
- **Containerization**: Docker

## 🚀 Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Groq API key (for LLM access)

### Environment Variables
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key
MYSQL_HOST=MYSQL_HOST
MYSQL_USER=MYSQL_USER
MYSQL_PASSWORD=MYSQL_PASSWORD
MYSQL_DATABASE=MYSQL_DATABASE i.e. academic_chatbot
```

### Running with Docker Compose
1. Clone the repository
```bash
git clone https://github.com/Japjeet09/Academic_chatbot.git
cd Academic_chatbot
```

2. Start the application
```bash
docker-compose up -d
```

3. Access the application at http://localhost:8510

### Running Locally with Virtual Environment
1. Clone the repository
```bash
git clone https://github.com/Japjeet09/Academic_chatbot.git
cd Academic_chatbot
```
2. Create and activate Virtual Environment (For Windows)
```bash
python -m venv venv
venv/Scripts/activate
```

3. Install packages
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
streamlit run app.py
```

5. Access the application at http://localhost:8510

## 💻 Usage Guide

### First-time Setup
1. Register a new account
2. Add subjects relevant to your studies
3. Upload syllabus files or manually add topics
4. Start chatting with the AI about your academic topics

### Exam Preparation
1. Navigate to the Exam Mode tab
2. Upload previous year question papers
3. Generate quizzes based on specific topics
4. Track your performance and review weak areas
5. Generate probable questions for exam preparation

## 📊 Architecture

```
+---------------+     +---------------+     +---------------+
|               |     |               |     |               |
|   Streamlit   |<--->|     MySQL     |     |    Qdrant     |
|   Frontend    |     |   Database    |     |Vector Database|
|               |     |               |     |               |
+---------------+     +---------------+     +---------------+
        ^                     ^                     ^
        |                     |                     |
        v                     v                     v
+----------------------------------------------------------+
|                                                          |
|                    Application Logic                     |
|                                                          |
+----------------------------------------------------------+
        ^                     ^                     ^
        |                     |                     |
        v                     v                     v
+---------------+     +---------------+     +---------------+
|               |     |               |     |               |
|    EasyOCR    |     |  Speech Rec.  |     |   Groq LLM    |
|               |     |               |     |               |
+---------------+     +---------------+     +---------------+
```

## 🤝 Contributions

Contributions are welcome! Please feel free to submit a Pull Request.
