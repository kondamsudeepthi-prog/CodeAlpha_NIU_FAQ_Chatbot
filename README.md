# CodeAlpha_NIU_FAQ_Chatbot

## 📌 Project Overview

An NLP-based FAQ chatbot developed as part of the CodeAlpha Artificial Intelligence Internship.

The chatbot is designed to answer frequently asked questions related to Noida International University (NIU) by matching user queries with the most relevant questions from an FAQ dataset.

The project demonstrates the practical application of Natural Language Processing, text similarity, information retrieval, and Streamlit-based application development.

## 🎯 Internship Task

**Internship:** CodeAlpha Artificial Intelligence Internship

**Domain:** Artificial Intelligence

**Task:** Task 2 – Chatbot for FAQs

The project follows the requirements provided by CodeAlpha for developing an FAQ chatbot.

The task includes:

- Collecting FAQs related to a specific topic
- Preprocessing user questions using NLP techniques
- Matching user questions with the most similar FAQ
- Displaying the best matching answer
- Providing a simple interface for user interaction

##  Key Features

- Interactive FAQ chatbot interface
- NIU-specific FAQ dataset
- Natural Language Processing-based text preprocessing
- TF-IDF text vectorization
- Cosine similarity-based question matching
- Exact question matching
- Synonym and question variation routing
- Relevance threshold for response selection
- Handles different variations of user queries
- Streamlit-based web interface
- Organized FAQ dataset for efficient retrieval

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application and chatbot interface |
| Pandas | Dataset loading and processing |
| NLTK | Natural Language Processing |
| Scikit-learn | Machine learning and similarity utilities |
| TF-IDF | Text vectorization |
| Cosine Similarity | FAQ similarity matching |
| Regular Expressions | Text normalization |

## 🧠 How It Works

The chatbot follows an NLP-based question-answering workflow.

```text
User Question
      ↓
Text Normalization
      ↓
NLP Preprocessing
      ↓
FAQ Question Matching
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Best Matching FAQ
      ↓
Similarity Threshold Check
      ↓
Chatbot Response
```

### 1. User Input

The user enters a question related to Noida International University through the chatbot interface.

### 2. Text Normalization

The input question is normalized by cleaning the text and handling variations in formatting and punctuation.

### 3. FAQ Matching

The processed question is compared with questions stored in the NIU FAQ dataset.

### 4. TF-IDF Vectorization

TF-IDF is used to convert textual questions into numerical feature vectors.

### 5. Similarity Calculation

Cosine similarity is used to determine how closely the user's question matches the available FAQ questions.

### 6. Best Match Selection

The FAQ with the highest relevant similarity score is selected as the potential answer.

### 7. Response Generation

The corresponding answer from the FAQ dataset is displayed to the user.

## 📊 Dataset

The chatbot uses an FAQ dataset containing questions, answers and category related to Noida International University.

The dataset covers topics including:

- University Information
- Admissions
- Courses
- Fees
- Scholarships
- Hostel
- Transportation
- Academics
- Examinations
- Placements
- Internships
- NIIMS
- Online programmes
- Campus facilities
  

## 🔍 Example Queries

The chatbot is designed to handle different variations of similar questions.

**Example:**

```text
Where can I find examination schedules?
```

```text
Where can I check exam timetable?
```

```text
What are the examination schedules?
```

Configured question routing allows different variations of a query to be directed to the appropriate FAQ when an exact or predefined equivalent is available.

## 🖥️ Application

The chatbot provides a simple Streamlit-based interface where users can:

- Enter an NIU-related question
- Submit the question
- Receive the most relevant FAQ answer
- View the matching result and relevance score where applicable

## 📂 Project Structure

```text
CodeAlpha_NIU_FAQ_Chatbot/
│
├── app.py
│
├── data/
│   └── faqs.csv
│
├── README.md
│
└── requirements.txt
```

## 💻 Installation

### Clone the Repository

```bash
git clone https://github.com/kondamsudeepthi-prog/CodeAlpha_NIU_FAQ_Chatbot.git
```

### Navigate to the Project Directory

```bash
cd CodeAlpha_NIU_FAQ_Chatbot
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in the browser through the Streamlit local server.

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing
- Text preprocessing
- TF-IDF vectorization
- Cosine similarity
- Information retrieval
- FAQ chatbot development
- Python application development
- Streamlit
- Dataset preparation
- Question matching
- Debugging and testing
- Git and GitHub
- Project version control

## 🚀 Future Improvements

The chatbot can be further enhanced with:

- Semantic embeddings for improved question understanding
- Transformer-based question matching
- Multilingual support
- Voice input and output
- Conversation history
- Retrieval-Augmented Generation (RAG)
- Improved intent classification
- Automated FAQ updates
- Production-ready deployment
  
## 👩‍💻 Author

** Kondam Sudeepthi **

B.Tech CSE (Artificial Intelligence & Machine Learning)

## 📜 License

This project is created for educational and internship purposes.
