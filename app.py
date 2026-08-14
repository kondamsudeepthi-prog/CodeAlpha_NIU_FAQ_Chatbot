import re
import pandas as pd
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NIU FAQ Chatbot",
    page_icon="🎓",
    layout="centered"
)

# =========================================================
# NLTK RESOURCES
# =========================================================

@st.cache_resource
def load_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords"]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    return set(stopwords.words("english"))


STOP_WORDS = load_nltk_resources()

# =========================================================
# LOAD UPDATED NIU DATASET
# =========================================================

DATASET_PATH = "Data/faqs.csv"


def load_faq_dataset():
    df = pd.read_csv(DATASET_PATH)

    required_columns = {"question", "answer", "category"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Remove empty values
    df = df.dropna(
        subset=["question", "answer", "category"]
    ).copy()

    # Standardize text
    for column in ["question", "answer", "category"]:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Remove empty questions/answers
    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ]

    # Remove exact duplicate questions
    df = df.drop_duplicates(
        subset=["question"],
        keep="first"
    )

    return df.reset_index(drop=True)


faq_df = load_faq_dataset()

# =========================================================
# TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):
    """Normalize common NIU terms so different user wording matches the FAQ."""
    text = str(text).lower()

    replacements = [
        # Education level
        ("10 + 2", "12"),
        ("10+2", "12"),
        ("class 12", "12"),
        ("12th standard", "12"),
        ("12th class", "12"),
        ("12th", "12"),
        ("twelfth", "12"),

        # Programme abbreviations
        ("b.tech", "btech"),
        ("b tech", "btech"),
        ("m.tech", "mtech"),
        ("m tech", "mtech"),
        ("b.pharm", "bpharm"),
        ("b pharm", "bpharm"),
        ("m.pharm", "mpharm"),
        ("m pharm", "mpharm"),
        ("b.com", "bcom"),
        ("b com", "bcom"),
        ("m.com", "mcom"),
        ("m com", "mcom"),
        ("b.sc", "bsc"),
        ("b sc", "bsc"),
        ("m.sc", "msc"),
        ("m sc", "msc"),

        # Course names / variants
        ("computer science and engineering", "cse"),
        ("computer science engineering", "cse"),
        ("artificial intelligence and machine learning", "ai ml"),
        ("ai & ml", "ai ml"),
        ("ai and ml", "ai ml"),
        ("data science and analytics", "data science analytics"),

        # Common singular/plural variants
        ("admissions", "admission"),
        ("requirements", "requirement"),
        ("qualifications", "qualification"),
        ("fees", "fee"),
        ("scholarships", "scholarship"),
        ("courses", "course"),
        ("programmes", "programme"),
        ("placements", "placement"),
        ("internships", "internship"),

        # NIU-specific terms
        ("erp", "erp"),
        ("lms", "lms"),
        ("niims", "niims"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    # Keep letters, numbers and spaces.
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    tokens = word_tokenize(text)

    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)


faq_df["processed_question"] = faq_df[
    "question"
].apply(preprocess_text)


# =========================================================
# TF-IDF MODEL
# =========================================================

@st.cache_resource
def create_tfidf_model(processed_questions):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.98
    )

    faq_vectors = vectorizer.fit_transform(
        processed_questions
    )

    return vectorizer, faq_vectors


vectorizer, faq_vectors = create_tfidf_model(
    tuple(faq_df["processed_question"])
)

# =========================================================
# SEMANTIC MODEL
# =========================================================

@st.cache_resource
def load_semantic_model():
    if not SEMANTIC_AVAILABLE:
        return None
    return SentenceTransformer("all-MiniLM-L6-v2")


semantic_model = load_semantic_model()

if semantic_model is not None:
    @st.cache_resource
    def create_semantic_vectors(questions):
        return semantic_model.encode(
            list(questions),
            normalize_embeddings=True,
            show_progress_bar=False
        )

    semantic_vectors = create_semantic_vectors(
        tuple(faq_df["question"])
    )
else:
    semantic_vectors = None


# =========================================================
# INTENT-AWARE RETRIEVAL
# =========================================================

INTENT_KEYWORDS = {
    "Fees & Scholarships": [
        "fee", "fees", "cost", "price", "tuition", "pay", "payment",
        "charges", "amount", "how much", "scholarship", "scholarships"
    ],
    "Admissions & Eligibility": [
    "admission", "apply", "application", "eligibility", "eligible",
    "qualification", "qualifications", "requirement", "requirements",
    "criteria", "marks", "percentage", "12th", "class 12", "pcm",
    "pcb", "entrance", "jee",

    # Natural admission/eligibility wording
    "after 12th", "after class 12", "after 10+2",
    "pass 12th", "passed 12th", "12th pass",
    "join btech", "join engineering",
    "enter btech", "enter engineering",
    "study btech", "study engineering",
    "can i do btech", "can i do engineering",
    "can i apply", "am i eligible",
    "minimum marks", "minimum percentage",
    "required marks", "required percentage",
    "pcm student", "12th pcm"
   ],
    "Courses & Programs": [
        "course", "courses", "program", "programs", "programme",
        "programmes", "offer", "offers", "available", "availability",
        "study", "specialization", "specialisation", "btech", "mtech",
        "bca", "mca", "mba", "bsc", "msc", "bpharm", "engineering"
    ],
    "Placements & Internships": [
        "placement", "placements", "placed", "job", "jobs", "career",
        "careers", "recruiter", "recruiters", "recruitment",
        "internship", "internships", "package", "packages", "salary"
    ],
    "Hostel": [
        "hostel", "accommodation", "stay", "staying", "residence",
        "room", "rooms", "mess", "cafeteria", "campus accommodation"
    ],
    "Transport": [
        "transport", "transportation", "bus", "buses", "route", "routes"
    ],
    "Medical / NIIMS": [
        "niims", "hospital", "medical", "health", "ambulance", "icu",
        "doctor", "doctors", "emergency", "radiology", "paediatrics"
    ],
    "Online / ERP / LMS": [
        "online", "erp", "lms", "portal", "login"
    ],
    "Examinations & Academics": [
        "exam", "examination", "semester", "marks", "attendance",
        "result", "results", "academic", "academics"
    ],
    "Documents & Certificates": [
        "document", "documents", "certificate", "certificates",
        "marksheet", "migration", "transfer", "id proof"
    ],
    "Contact Information": [
        "contact", "phone", "number", "email", "address", "helpline",
        "registrar"
    ],
    "Campus & Facilities": [
        "campus", "library", "facility", "facilities", "gym", "sports",
        "wifi"
    ],
    "Student Services": [
        "student council", "club", "clubs", "event", "events",
        "activity", "activities"
    ]
}

# Phrases that often express the same intent but use different words.
# ============================================================
# QUERY EXPANSIONS
# ============================================================

QUERY_EXPANSIONS = {

    # ========================================================
    # ADMISSIONS - GENERAL
    # ========================================================

    "admission": [
        "admission",
        "admissions",
        "admission process",
        "admission procedure",
        "how to get admission",
        "how can I get admission",
        "how to apply for admission",
        "application process",
        "admission requirements",
        "admission criteria",
        "eligibility for admission",
        "NIU admission",
        "university admission",
        "college admission"
    ],

    "apply": [
        "apply",
        "application",
        "apply for admission",
        "how to apply",
        "application process",
        "admission application",
        "online application",
        "apply to NIU",
        "apply for NIU",
        "how can I apply"
    ],

    "eligibility": [
        "eligibility",
        "eligible",
        "eligibility criteria",
        "admission eligibility",
        "qualification required",
        "academic qualification",
        "minimum qualification",
        "admission requirements",
        "eligibility requirements",
        "who is eligible",
        "am I eligible",
        "can I apply"
    ],

    # ========================================================
    # B.TECH GENERAL
    # ========================================================

    "btech": [
        "B.Tech",
        "BTech",
        "B Tech",
        "Bachelor of Technology",
        "engineering",
        "engineering degree",
        "engineering course",
        "technical degree",
        "undergraduate engineering",
        "B.Tech programme",
        "B.Tech course",
        "B.Tech admission"
    ],

    "btech eligibility": [
        "B.Tech eligibility",
        "BTech eligibility",
        "B Tech eligibility",
        "eligibility for B.Tech",
        "B.Tech admission eligibility",
        "B.Tech eligibility criteria",
        "B.Tech admission requirements",
        "qualification required for B.Tech",
        "academic qualification for B.Tech",
        "minimum qualification for B.Tech",
        "engineering eligibility",
        "engineering admission eligibility",
        "eligibility for engineering",
        "minimum qualification for engineering",
        "B.Tech after 12th",
        "engineering after 12th"
    ],

    "btech after 12th": [
        "B.Tech after 12th",
        "B.Tech after Class 12",
        "B.Tech after 10+2",
        "engineering after 12th",
        "engineering after Class 12",
        "engineering after 10+2",
        "can I do B.Tech after 12th",
        "can I join B.Tech after 12th",
        "can I study engineering after 12th",
        "12th pass B.Tech admission",
        "12th pass engineering admission",
        "B.Tech eligibility after 12th",
        "B.Tech eligibility after 10+2"
    ],

    # ========================================================
    # PCM
    # ========================================================

    "pcm": [
        "PCM",
        "Physics Chemistry Mathematics",
        "Physics Mathematics Chemistry",
        "12th PCM",
        "Class 12 PCM",
        "10+2 PCM",
        "PCM student",
        "PCM students",
        "PCM eligibility",
        "PCM eligibility for B.Tech",
        "PCM eligibility for engineering",
        "B.Tech after PCM",
        "engineering after PCM",
        "B.Tech after 12th PCM",
        "engineering after 12th PCM",
        "can PCM students apply for B.Tech",
        "can PCM student join engineering",
        "PCM student admission"
    ],

    "pcm engineering": [
        "PCM for engineering",
        "PCM required for engineering",
        "PCM eligibility for engineering",
        "PCM subjects for engineering",
        "Physics Chemistry Mathematics for engineering",
        "Physics Maths Chemistry for engineering",
        "12th PCM engineering admission",
        "PCM student engineering admission",
        "can PCM students do engineering",
        "can PCM student join engineering",
        "engineering after 12th PCM"
    ],

    "pcm btech": [
        "PCM for B.Tech",
        "PCM eligibility for B.Tech",
        "PCM student B.Tech",
        "B.Tech after PCM",
        "B.Tech after 12th PCM",
        "12th PCM B.Tech admission",
        "PCM student B.Tech admission",
        "can PCM student apply for B.Tech",
        "can PCM student get B.Tech admission",
        "B.Tech admission for PCM students"
    ],

    # ========================================================
    # 12TH / 10+2
    # ========================================================

    "12th": [
        "12th",
        "Class 12",
        "12th standard",
        "10+2",
        "higher secondary",
        "senior secondary",
        "12th pass",
        "passed Class 12",
        "completed 12th",
        "after 12th",
        "after Class 12",
        "after 10+2"
    ],

    "12th eligibility": [
        "12th eligibility for B.Tech",
        "Class 12 eligibility for B.Tech",
        "10+2 eligibility for B.Tech",
        "12th qualification for B.Tech",
        "Class 12 qualification for engineering",
        "10+2 qualification for engineering",
        "12th pass B.Tech eligibility",
        "12th pass engineering eligibility",
        "B.Tech eligibility after 12th",
        "engineering eligibility after 12th"
    ],

    "passed 12th": [
        "I passed 12th",
        "I passed Class 12",
        "I completed 12th",
        "I completed Class 12",
        "12th pass student",
        "12th passed student",
        "passed 10+2",
        "completed 10+2",
        "12th pass B.Tech",
        "12th pass engineering",
        "can I join B.Tech after passing 12th",
        "can I join engineering after passing 12th"
    ],

    # ========================================================
    # SUBJECTS
    # ========================================================

    "subjects": [
        "subjects required for B.Tech",
        "subjects required for engineering",
        "subjects required in Class 12 for B.Tech",
        "subjects required in 10+2 for B.Tech",
        "subjects needed for engineering",
        "Class 12 subjects for engineering",
        "12th subjects required for B.Tech",
        "what subjects should I study for B.Tech",
        "what subjects are required for B.Tech",
        "what should I study in Class 12",
        "subjects needed for B.Tech admission"
    ],

    "physics mathematics chemistry": [
        "Physics Mathematics Chemistry",
        "Physics Maths Chemistry",
        "Physics Chemistry Mathematics",
        "PCM subjects",
        "PCM combination",
        "Class 12 Physics Mathematics Chemistry",
        "12th Physics Maths Chemistry",
        "Physics Maths Chemistry for B.Tech",
        "Physics Mathematics Chemistry for engineering"
    ],

    # ========================================================
    # PERCENTAGE / MARKS
    # ========================================================

    "percentage": [
        "percentage",
        "marks",
        "minimum percentage",
        "minimum marks",
        "required percentage",
        "required marks",
        "percentage required",
        "marks required",
        "minimum percentage for B.Tech",
        "minimum marks for B.Tech",
        "minimum percentage for engineering",
        "minimum marks for engineering",
        "12th percentage required",
        "12th marks required",
        "10+2 percentage required",
        "10+2 marks required"
    ],

    "minimum percentage": [
        "minimum percentage for B.Tech",
        "minimum percentage required for B.Tech",
        "minimum percentage for engineering",
        "minimum percentage required for engineering",
        "minimum marks for B.Tech",
        "minimum marks required for B.Tech",
        "minimum marks for engineering",
        "minimum marks required for engineering",
        "12th minimum percentage",
        "12th minimum marks",
        "10+2 minimum percentage",
        "10+2 minimum marks",
        "percentage requirement for B.Tech",
        "percentage requirement for engineering"
    ],

    "50 percent": [
        "50 percent",
        "50%",
        "minimum 50 percent",
        "minimum 50% marks",
        "50 percent marks",
        "50% marks",
        "50 percent in 12th",
        "50% in 12th",
        "50 percent in 10+2",
        "50% in 10+2",
        "B.Tech eligibility 50%",
        "engineering eligibility 50%",
        "B.Tech admission with 50 percent",
        "can I apply with 50 percent"
    ],

    "marks": [
        "marks required for B.Tech",
        "minimum marks for B.Tech",
        "marks needed for engineering",
        "minimum marks for engineering",
        "12th marks for B.Tech",
        "12th marks required",
        "10+2 marks required",
        "marks requirement for admission",
        "minimum qualifying marks"
    ],

    # ========================================================
    # SPECIFIC PERCENTAGE QUERIES
    # ========================================================

    "52 percent": [
        "52% in 12th",
        "52 percent in 12th",
        "I got 52% in 12th",
        "I got 52 percent in 12th",
        "52% PCM",
        "52 percent PCM",
        "52% B.Tech eligibility",
        "can I apply with 52%",
        "can I get B.Tech admission with 52%"
    ],

    "60 percent": [
        "60% in 12th",
        "60 percent in 12th",
        "I got 60% in 12th",
        "60% B.Tech eligibility",
        "can I apply with 60%"
    ],

    "70 percent": [
        "70% in 12th",
        "70 percent in 12th",
        "I got 70% in 12th",
        "70% B.Tech eligibility",
        "can I apply with 70%"
    ],

    # ========================================================
    # ENGINEERING
    # ========================================================

    "engineering": [
        "engineering",
        "engineering course",
        "engineering programme",
        "engineering degree",
        "B.Tech engineering",
        "engineering admission",
        "engineering eligibility",
        "engineering qualification",
        "engineering requirements",
        "study engineering at NIU",
        "join engineering at NIU"
    ],

    "engineering eligibility": [
        "engineering eligibility",
        "engineering admission eligibility",
        "eligibility for engineering",
        "engineering qualification",
        "engineering admission requirements",
        "minimum qualification for engineering",
        "minimum marks for engineering",
        "minimum percentage for engineering",
        "12th eligibility for engineering",
        "10+2 eligibility for engineering",
        "B.Tech engineering eligibility"
    ],

    # ========================================================
    # COURSES / PROGRAMMES
    # ========================================================

    "course": [
        "course",
        "courses",
        "programme",
        "programmes",
        "degree programme",
        "academic programme",
        "courses offered",
        "programmes offered",
        "courses available",
        "programmes available",
        "courses at NIU",
        "programmes at NIU"
    ],

    "btech courses": [
        "B.Tech courses",
        "B.Tech programmes",
        "BTech courses",
        "B.Tech branches",
        "engineering courses",
        "engineering branches",
        "engineering programmes",
        "B.Tech specializations",
        "B.Tech programmes at NIU",
        "engineering programmes at NIU"
    ],

    "ai ml": [
        "AI ML",
        "AI and ML",
        "Artificial Intelligence and Machine Learning",
        "Artificial Intelligence",
        "Machine Learning",
        "B.Tech AI ML",
        "B.Tech Artificial Intelligence",
        "B.Tech Machine Learning",
        "B.Tech CSE AI ML",
        "Computer Science Engineering AI ML"
    ],

    # ========================================================
    # FEES
    # ========================================================

    "fees": [
        "fees",
        "fee",
        "tuition fee",
        "course fee",
        "academic fee",
        "annual fee",
        "semester fee",
        "B.Tech fees",
        "B.Tech fee",
        "engineering fees",
        "engineering fee",
        "programme fee",
        "total fees",
        "fee structure",
        "fees structure",
        "fee details"
    ],

    "btech fees": [
        "B.Tech fees",
        "B.Tech fee structure",
        "B.Tech tuition fee",
        "B.Tech course fee",
        "B.Tech annual fee",
        "B.Tech semester fee",
        "engineering fees",
        "engineering fee structure",
        "B.Tech total fees",
        "B.Tech fee details"
    ],

    # ========================================================
    # SCHOLARSHIPS
    # ========================================================

    "scholarship": [
        "scholarship",
        "scholarships",
        "scholarship eligibility",
        "scholarship criteria",
        "scholarship requirements",
        "financial aid",
        "financial assistance",
        "student scholarship",
        "NIU scholarship",
        "scholarships at NIU",
        "merit scholarship",
        "scholarship application",
        "how to get scholarship"
    ],

    # ========================================================
    # HOSTEL
    # ========================================================

    "hostel": [
        "hostel",
        "hostels",
        "hostel facility",
        "hostel facilities",
        "hostel accommodation",
        "accommodation",
        "student accommodation",
        "hostel rooms",
        "hostel fee",
        "hostel charges",
        "hostel amenities",
        "hostel at NIU",
        "stay at university",
        "can I stay at university",
        "residential facility"
    ],

    "hostel fees": [
        "hostel fees",
        "hostel fee",
        "hostel charges",
        "hostel accommodation fee",
        "hostel cost",
        "hostel pricing",
        "residential fee"
    ],

    # ========================================================
    # TRANSPORT
    # ========================================================

    "transport": [
        "transport",
        "transportation",
        "transport facility",
        "transport facilities",
        "university transport",
        "college transport",
        "bus facility",
        "bus service",
        "university bus",
        "college bus",
        "transportation facility",
        "campus transport"
    ],
        "bus services": [
        "transport",
        "bus",
        "transport facilities",
        "transport services"
    ],

    "bus service": [
        "transport",
        "bus",
        "transport facilities",
        "transport services"
    ],

    "buses available": [
        "transport",
        "bus",
        "transport facilities"
    ],

    "bus available": [
        "transport",
        "bus",
        "transport facilities"
    ],

    "bus facility": [
        "transport",
        "bus",
        "transport facilities"
    ],
    # ========================================================
    # ACADEMICS
    # ========================================================

    "academics": [
        "academics",
        "academic",
        "academic facilities",
        "academic support",
        "teaching",
        "teaching facilities",
        "classes",
        "lectures",
        "faculty",
        "professors",
        "teachers",
        "academic programmes",
        "academic system"
    ],

    # ========================================================
    # EXAMINATIONS
    # ========================================================

    "exams": [
        "exam",
        "exams",
        "examination",
        "examinations",
        "semester exams",
        "end semester exams",
        "internal exams",
        "assessment",
        "examination process",
        "exam pattern",
        "exam schedule",
        "evaluation"
    ],

    # ========================================================
    # PLACEMENTS
    # ========================================================

    "placements": [
        "placement",
        "placements",
        "campus placement",
        "campus placements",
        "placement opportunities",
        "placement support",
        "career support",
        "career opportunities",
        "jobs after graduation",
        "graduate jobs",
        "placement assistance",
        "placement cell",
        "companies visiting campus",
        "recruiters"
    ],

    "jobs": [
        "jobs after graduation",
        "job opportunities",
        "career opportunities",
        "employment opportunities",
        "graduate jobs",
        "placement opportunities",
        "how does NIU help graduates find jobs",
        "career support",
        "placement support"
    ],

    # ========================================================
    # INTERNSHIPS
    # ========================================================

    "internships": [
        "internship",
        "internships",
        "internship opportunities",
        "internship support",
        "student internships",
        "industry internships",
        "summer internship",
        "internship programme",
        "internship opportunities at NIU",
        "does NIU provide internships"
    ],

    # ========================================================
    # NIIMS
    # ========================================================

    "niims": [
        "NIIMS",
        "Noida International Institute of Medical Sciences",
        "medical institute",
        "medical college",
        "NIIMS hospital",
        "NIIMS facilities",
        "NIIMS courses",
        "NIIMS admission"
    ],

    # ========================================================
    # ONLINE PROGRAMMES
    # ========================================================

    "online": [
        "online programme",
        "online programmes",
        "online course",
        "online courses",
        "online degree",
        "online education",
        "distance learning",
        "online learning",
        "online programme at NIU",
        "online courses at NIU"
    ],

    # ========================================================
    # CAMPUS FACILITIES
    # ========================================================

    "campus facilities": [
        "campus facilities",
        "campus facility",
        "university facilities",
        "college facilities",
        "student facilities",
        "campus amenities",
        "university amenities",
        "facilities at NIU",
        "facilities available at NIU"
    ],

    # ========================================================
    # LIBRARY
    # ========================================================

    "library": [
        "library",
        "library facility",
        "library facilities",
        "university library",
        "college library",
        "books",
        "digital library",
        "library resources",
        "library at NIU"
    ],

    # ========================================================
    # SPORTS / GYM
    # ========================================================

    "sports": [
        "sports",
        "sports facilities",
        "sports facility",
        "sports activities",
        "games",
        "athletics",
        "indoor games",
        "outdoor games",
        "sports at NIU"
    ],

    "gym": [
        "gym",
        "fitness centre",
        "fitness center",
        "gym facility",
        "fitness facilities",
        "workout facility",
        "gym at hostel",
        "hostel gym"
    ],

    # ========================================================
    # CAMPUS / LOCATION
    # ========================================================

    "campus": [
        "campus",
        "university campus",
        "college campus",
        "NIU campus",
        "campus location",
        "where is NIU",
        "NIU location",
        "university location"
    ],

    "location": [
        "location",
        "address",
        "university address",
        "college address",
        "NIU address",
        "where is NIU located",
        "where is the university",
        "campus location"
    ],

    # ========================================================
    # TRAVEL / HOW TO REACH
    # ========================================================

    "travel": [
        "travel to NIU",
        "how to reach NIU",
        "how can I reach NIU",
        "how to travel to NIU",
        "transport to NIU",
        "reach university",
        "reach campus",
        "directions to NIU",
        "route to NIU"
    ],

    # ========================================================
    # ADMISSION + PCM COMBINATIONS
    # ========================================================

    "pcm percentage": [
        "PCM percentage required for B.Tech",
        "minimum percentage for PCM students",
        "12th PCM percentage for B.Tech",
        "minimum PCM marks for engineering",
        "PCM student with 50 percent B.Tech",
        "PCM student eligibility for B.Tech",
        "B.Tech eligibility percentage for PCM",
        "engineering eligibility percentage PCM",
        "PCM marks required for B.Tech",
        "PCM percentage for engineering"
    ],

    "pcm 12th": [
        "12th PCM B.Tech",
        "Class 12 PCM B.Tech",
        "10+2 PCM B.Tech",
        "PCM after 12th",
        "B.Tech after 12th PCM",
        "engineering after 12th PCM",
        "12th PCM student eligibility",
        "12th PCM admission"
    ],

    # ========================================================
    # GENERIC QUESTIONS
    # ========================================================

    "can i": [
        "can I apply",
        "can I join",
        "can I get admission",
        "am I eligible",
        "is it possible",
        "eligibility",
        "admission eligibility",
        "admission requirements"
    ],

    "required": [
        "required",
        "requirements",
        "eligibility requirements",
        "admission requirements",
        "minimum requirements",
        "qualification required",
        "documents required",
        "marks required",
        "percentage required"
    ]
}


def expand_query(text):
    text = str(text).lower()

    expanded = [text]

    for phrase, replacement in QUERY_EXPANSIONS.items():

        if phrase in text:

            # If expansion is a list, add all phrases
            if isinstance(replacement, list):
                expanded.extend(replacement)

            # If expansion is a string, add it directly
            else:
                expanded.append(replacement)

    return " ".join(expanded)

def detect_category(user_question):
    text = str(user_question).lower()
    scores = {}

    for category, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                # Longer phrases are stronger signals.
                score += 2 if " " in keyword else 1
        scores[category] = score

    best_category = max(scores, key=scores.get)
    return best_category if scores[best_category] > 0 else None

def detect_intent(user_question):
    text = str(user_question).lower()

    if any(k in text for k in INTENT_KEYWORDS["Fees & Scholarships"]):
        return "Fees & Scholarships"
    if any(k in text for k in INTENT_KEYWORDS["Admissions & Eligibility"]):
        return "Admissions & Eligibility"
    if any(k in text for k in INTENT_KEYWORDS["Placements & Internships"]):
        return "Placements & Internships"
    if any(k in text for k in INTENT_KEYWORDS["Hostel"]):
        return "Hostel"
    if any(k in text for k in INTENT_KEYWORDS["Transport"]):
        return "Transport"
    if any(k in text for k in INTENT_KEYWORDS["Medical / NIIMS"]):
        return "Medical / NIIMS"
    if any(k in text for k in INTENT_KEYWORDS["Online / ERP / LMS"]):
        return "Online / ERP / LMS"
    if any(k in text for k in INTENT_KEYWORDS["Courses & Programs"]):
        return "Courses & Programs"
    return detect_category(user_question)

def find_best_match(user_question):
    """Hybrid + TF-IDF + semantic + lexical FAQ retrieval."""

    # =========================================================
    # EXAMINATION EXACT MATCH ROUTER
    # =========================================================

    query = str(user_question).lower().strip()

    # Remove punctuation
    query_clean = re.sub(r"[^a-z0-9\s]", "", query)
    query_clean = re.sub(r"\s+", " ", query_clean).strip()

    # ---------------------------------------------------------
    # Exact examination question mapping
    # ---------------------------------------------------------

    exam_mapping = {

        "where can i find examination schedules":
            "Where can I find NIU examination schedules?",

        "where can i find the examination schedules":
            "Where can I find NIU examination schedules?",

        "what are the examination schedules":
            "Where can I find NIU examination schedules?",

        "what are the examination schedule":
            "Where can I find NIU examination schedules?",

        "where can i check exam timetable":
            "Where can I find NIU examination schedules?",

        "where can i check the exam timetable":
            "Where can I find NIU examination schedules?",

        "where can i check examination timetable":
            "Where can I find NIU examination schedules?",

        "where can i find examination notices":
            "Where can I find examination notices?",

        "where can i find exam notices":
            "Where can I find examination notices?",

        "does niu publish examination notices online":
            "Does NIU publish examination notices online?",

        "does niu publish exam notices online":
            "Does NIU publish examination notices online?",
    }

    # =========================================================
    # CHECK EXACT EXAM QUESTION
    # =========================================================

    if query_clean in exam_mapping:

        target_question = exam_mapping[query_clean]

        # Normalize dataset questions
        questions = (
            faq_df["question"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        target_clean = (
            target_question.lower()
            .strip()
        )

        # Exact question match
        match = faq_df[
            questions == target_clean
        ]

        if not match.empty:

            row = match.iloc[0]

            print("🔥 EXAM EXACT MATCH")
            print("🔥 USER:", user_question)
            print("🔥 DATASET QUESTION:", row["question"])
            print("🔥 ANSWER:", row["answer"])

            return match.index[0], 1.0

    # =========================================================
    # EXACT BUS TIMINGS FAQ
    # =========================================================

    raw_query = str(user_question).strip().lower()

    if (
        "bus" in raw_query
        and (
            "timing" in raw_query
            or "timings" in raw_query
        )
    ):
        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):
            raw_faq = str(
                faq_question
            ).strip().lower()

            if (
                "publish bus timing" in raw_faq
                or "bus timing" in raw_faq
            ):
                return i, 1.0
    
    
    

    # =========================================================
    # 1. QUERY EXPANSION
    # =========================================================

    expanded_query = expand_query(user_question)
    # =========================================================
    # 1. QUERY EXPANSION
    # =========================================================
    expanded_query = expand_query(user_question)

    # Safety: expand_query list return chesina kuda handle chestundi
    if isinstance(expanded_query, list):
        expanded_query = " ".join(
            str(x) for x in expanded_query
            if x is not None
        )

    processed_query = preprocess_text(expanded_query)

    if not processed_query:
        return None, 0.0

    
    

    # =========================================================
    # EXAMINATION FAQ - EXACT/SYNONYM ROUTER
    # =========================================================

    raw_query = user_question.lower().strip()

    # Normalize punctuation
    normalized_query = re.sub(r"[^a-z0-9\s]", "", raw_query)
    normalized_query = re.sub(r"\s+", " ", normalized_query).strip()

    exam_routes = {
        # Examination schedule
        "what is the exam schedule": "where can i find niu examination schedules",
        "what are the exam schedules": "where can i find niu examination schedules",
        "what is the examination schedule": "where can i find niu examination schedules",
        "what are the examination schedules": "where can i find niu examination schedules",
        "where can i find examination schedules": "where can i find niu examination schedules",
        "where can i find the examination schedules": "where can i find niu examination schedules",
        "where can i check exam timetable": "where can i find niu examination schedules",
        "where can i check the exam timetable": "where can i find niu examination schedules",
        "where can i check examination timetable": "where can i find niu examination schedules",
        
        # Examination notices
        "does niu publish examination notices online": "does niu publish examination notices online",
        "does niu publish exam notices online": "does niu publish examination notices online",
        "are examination notices published online": "does niu publish examination notices online",
        "are exam notices published online": "does niu publish examination notices online",
        
        # Examination notice location
        "where can i find examination notices": "where can i find examination notices",
        "where can i find exam notices": "where can i find examination notices",
    }

    if normalized_query in exam_routes:
        
        print("🔥 EXAM ROUTE MATCHED:", normalized_query)

        target_question = exam_routes[normalized_query]

        print("🔥 TARGET QUESTION:", target_question)

        faq_questions_clean = (
            faq_df["question"]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        target_clean = re.sub(
            r"[^a-z0-9\s]",
            "",
            target_question.lower()
        )
        target_clean = re.sub(
            r"\s+",
            " ",
            target_clean
        ).strip()

        print("🔥 TARGET CLEAN:", target_clean)

        # Find examination schedule FAQ by important keywords
        matches = faq_df[
            faq_questions_clean.str.contains(
                "examination",
                na=False
            )
            &
            faq_questions_clean.str.contains(
                "schedule",
                na=False
            )
            &
            faq_questions_clean.str.contains(
                "find",
                na=False
            )
        ]

        print("🔥 SCHEDULE MATCH COUNT:", len(matches))
        print(
            "🔥 SCHEDULE MATCHED QUESTIONS:",
            matches["question"].tolist()
        )

        if not matches.empty:

            matched_index = matches.index[0]

            print(
                "🔥 EXACT EXAM FAQ:",
                faq_df.loc[matched_index, "question"]
            )

            print(
                "🔥 EXACT EXAM ANSWER:",
                faq_df.loc[matched_index, "answer"]
            )

            return matched_index, 1.0
            
            
        print("🔥 MATCH COUNT:", len(matches))
        print(
            "🔥 MATCHED QUESTIONS:",
            matches["question"].tolist()
        )
        
        

        if not matches.empty:

            matched_index = matches.index[0]

            print("🔥 EXACT EXAM ROUTER:", normalized_query)
            print("🔥 TARGET FAQ:", faq_df.loc[matched_index, "question"])
            print("🔥 TARGET ANSWER:", faq_df.loc[matched_index, "answer"])

            return matched_index, 1.0
        
    

        
        
       
    # =========================================================
    # 2. TF-IDF SIMILARITY
    # =========================================================
    query_vector = vectorizer.transform(
        [processed_query]
    )

    tfidf_scores = cosine_similarity(
        query_vector,
        faq_vectors
    )[0]

    # =========================================================
    # 3. SEMANTIC SIMILARITY
    # =========================================================
    if (
        semantic_model is not None
        and semantic_vectors is not None
    ):
        query_embedding = semantic_model.encode(
            [user_question],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        semantic_scores = cosine_similarity(
            query_embedding,
            semantic_vectors
        )[0]

    else:
        # If semantic model is unavailable,
        # use TF-IDF scores as fallback
        semantic_scores = tfidf_scores.copy()

    # =========================================================
    # 4. DETECT INTENT / CATEGORY
    # =========================================================
    intent = detect_intent(user_question)

    if intent:
        category_mask = (
            faq_df["category"].values == intent
        )
    else:
        category_mask = None

    # =========================================================
    # 5. LEXICAL OVERLAP
    # =========================================================
    query_tokens = set(
        processed_query.split()
    )

    lexical_scores = []

    for question in faq_df["processed_question"]:
        question_tokens = set(
            question.split()
        )

        overlap = len(
            query_tokens & question_tokens
        )

        lexical_scores.append(
            overlap / max(len(query_tokens), 1)
        )

    lexical_scores = pd.Series(
        lexical_scores
    ).values

    # =========================================================
    # 6. HYBRID RANKING
    # =========================================================
    adjusted_scores = (
        0.60 * semantic_scores
        + 0.30 * tfidf_scores
        + 0.10 * lexical_scores
    )

    # =========================================================
    # 7. CATEGORY CONSTRAINT
    # =========================================================
    if (
        category_mask is not None
        and category_mask.any()
    ):
        adjusted_scores[
            ~category_mask
        ] = -1.0

    
    # =========================================================
    # TRANSPORT INTENT ROUTER
    # =========================================================

    query = str(user_question).lower().strip()

    # ---------------------------------------------------------
    # 1. TRANSPORT TIMINGS
    # ---------------------------------------------------------

    raw_query = str(user_question).lower().strip()

    if (
        "bus" in raw_query
        and (
            "timing" in raw_query
            or "timings" in raw_query
        )
    ):

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):

            faq_question = str(
                faq_question
            ).lower().strip()

            # Target the specific bus-timing FAQ
            if (
                "publish" in faq_question
                and "bus" in faq_question
                and "timing" in faq_question
            ):
                return i, 1.0
        

    # ---------------------------------------------------------
    # 2. DELHI TRANSPORT
    # ---------------------------------------------------------

    if "delhi" in query:

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):
            faq_question = str(faq_question).lower()

            if (
                "transport" in faq_question
                and "delhi" in faq_question
            ):
                return i, 1.0


    # ---------------------------------------------------------
    # 3. GHAZIABAD TRANSPORT
    # ---------------------------------------------------------

    if "ghaziabad" in query:

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):
            faq_question = str(faq_question).lower()

            if (
                "transport" in faq_question
                and "ghaziabad" in faq_question
            ):
                return i, 1.0


    # ---------------------------------------------------------
    # 4. AREAS / ROUTES COVERED
    # ---------------------------------------------------------

    area_terms = [
        "what areas",
        "which areas",
        "areas covered",
        "areas are covered",
        "routes covered",
        "which routes",
        "what routes"
    ]

    if any(term in query for term in area_terms):

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):
            faq_question = str(faq_question).lower()

            if (
                "transport" in faq_question
                and (
                    "areas" in faq_question
                    or "routes" in faq_question
                )
            ):
                return i, 1.0


    # ---------------------------------------------------------
    # 5. GENERIC BUS / TRANSPORT FACILITY
    # ---------------------------------------------------------

    if (
        "bus" in query
        or "buses" in query
        or "transport" in query
    ):

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):
            faq_question = str(faq_question).lower()

            if (
                "transport" in faq_question
                and (
                    "provide transport facilities" in faq_question
                    or "run ac and non-ac buses" in faq_question
                    or "transport facilities" in faq_question
                )
            ):
                return i, 1.0

    # =========================================================
    # 8. PROGRAMME CONSISTENCY
    # =========================================================
    program_terms = [
        "btech",
        "bca",
        "mca",
        "mba",
        "bba",
        "bsc",
        "msc",
        "bpharm",
        "mtech",
        "cse",
        "ai ml",
        "cyber security",
        "data science",
        "biotechnology",
        "civil engineering",
        "mechanical engineering",
        "electrical engineering"
    ]

    query_programs = [
        term
        for term in program_terms
        if term in processed_query
    ]

    if query_programs:

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):

            if any(
                term in faq_question
                for term in query_programs
            ):
                adjusted_scores[i] += 0.10
    # =========================================================
    # B.TECH / ENGINEERING ELIGIBILITY BOOST  
    # =========================================================

    engineering_terms = [
        "btech",
        "b.tech",
        "engineering",
        "engineer",
        "pcm",
        "physics",
        "mathematics",
        "chemistry",
        "12th",
        "10+2"
    ]

    if any(term in processed_query for term in engineering_terms):

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):

            # Strongly prefer normal B.Tech / Engineering FAQs
            if (
                ("btech" in faq_question or
                "b tech" in faq_question or
                "engineering" in faq_question)
                and
                ("biotechnology" not in faq_question)
            ):
                adjusted_scores[i] += 0.15
    # =========================================================
    # B.TECH GENERIC QUERY BOOST
    # Avoid branch-specific answers for generic B.Tech questions
    # =========================================================

    btech_terms = [
        "btech",
        "b tech",
        "b.tech",
        "engineering"
    ]

    branch_terms = [
    "civil",
    "mechanical",
    "computer science",
    "cse",
    "electrical",
    "electronics",
    "ece",
    "ai ml",
    "artificial intelligence",
    "data science",
    "biotechnology",
    "chemical",
    "robotics",
    "robotic"
]
    

    is_btech_query = any(
        term in processed_query
        for term in btech_terms
    )

    has_branch = any(
        term in processed_query
        for term in branch_terms
    )

    if is_btech_query and not has_branch:

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):

            is_generic_btech_faq = (
                (
                    "btech" in faq_question
                    or "b tech" in faq_question
                    or "b.tech" in faq_question
                )
                and not any(
                    branch in faq_question
                    for branch in branch_terms
                )
            )

            is_branch_specific_faq = any(
                branch in faq_question
                for branch in branch_terms
            )

            if is_generic_btech_faq:
                adjusted_scores[i] += 0.20

            elif is_branch_specific_faq:
                adjusted_scores[i] -= 0.15   
    
    # =========================================================
    # B.TECH GENERIC FEE - EXACT ROUTER
    # =========================================================

    raw_query = str(user_question).lower().strip()

    is_generic_btech_fee = (
        ("btech" in raw_query or "b tech" in raw_query or "b.tech" in raw_query)
        and ("fee" in raw_query or "fees" in raw_query)
        and not any(
            branch in raw_query
            for branch in branch_terms
        )
    )

    if is_generic_btech_fee:

        for i, faq_question in enumerate(
            faq_df["question"]
        ):
            faq_question = str(
                faq_question
            ).lower().strip()

            if (
                "fee structure for b.tech at niu" in faq_question
                or "fee structure for btech at niu" in faq_question
            ):
                return i, 1.0
    # =========================================================
    # 9. INTENT-SPECIFIC CONSISTENCY
    # =========================================================
    intent_terms = {

        "Admissions & Eligibility": [
            "eligibility",
            "eligible",
            "requirement",
            "qualification",
            "marks",
            "percentage",
            "12",
            "admission",
            "apply"
        ],

        "Fees & Scholarships": [
            "fee",
            "fees",
            "cost",
            "tuition",
            "amount",
            "payment",
            "scholarship"
        ],

        "Placements & Internships": [
            "placement",
            "career",
            "job",
            "recruitment",
            "internship",
            "package"
        ],

        "Hostel": [
            "hostel",
            "accommodation",
            "room",
            "residence",
            "stay"
        ],

        "Transport": [
            "transport",
            "bus",
            "route"
        ],

        "Courses & Programs": [
            "course",
            "program",
            "programme",
            "specialization",
            "offer",
            "available"
        ]
    }

    if intent in intent_terms:

        for i, faq_question in enumerate(
            faq_df["processed_question"]
        ):

            overlap = sum(
                1
                for term in intent_terms[intent]
                if term in processed_query
                and term in faq_question
            )

            adjusted_scores[i] += (
                0.015 * overlap
            )
    # =========================================================
    # 10. BEST MATCH
    # =========================================================

    best_index = int(
        adjusted_scores.argmax()
    )

    best_score = float(
        adjusted_scores[best_index]
    )

    return best_index, best_score

# =========================================================
# CHATBOT RESPONSE
# =========================================================

# Controlled threshold after intent/programme reranking.
SIMILARITY_THRESHOLD = 0.48


def get_chatbot_response(user_question):
    best_index, score = find_best_match(
        user_question
    )

    if best_index is None:
        return (
            "Please enter a question related to "
            "Noida International University.",
            0.0
        )

    if score < SIMILARITY_THRESHOLD:
        return (
            "Sorry, I couldn't find a sufficiently "
            "relevant answer in the NIU FAQ dataset. "
            "Please try asking about admissions, "
            "courses, fees, scholarships, hostel, "
            "transport, academics, examinations, "
            "placements, internships, NIIMS, "
            "online programmes or campus facilities.",
            score
        )

    answer = faq_df.iloc[best_index]["answer"]

    return answer, score


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("🎓 NIU FAQ Chatbot")

st.write(
    "Ask questions about Noida International University "
    "including admissions, courses, fees, scholarships, "
    "hostel, transport, academics, examinations, "
    "placements, internships, NIIMS and online programmes."
)

st.caption(
    f"📚 FAQ Dataset: {len(faq_df)} verified entries"
)

st.divider()

# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])

        if (
            message["role"] == "assistant"
            and "score" in message
        ):
            st.caption(
                f"Match Score: "
                f"{message['score']:.2f}"
            )

# =========================================================
# USER INPUT
# =========================================================

user_question = st.chat_input(
    "Ask your question about NIU..."
)

if user_question:

    st.session_state.messages.append({
        "role": "user",
        "content": user_question
    })

    with st.chat_message("user"):
        st.write(user_question)

    answer, score = get_chatbot_response(
        user_question
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "score": score
    })

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(
            f"Match Score: {score:.2f}"
        )

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("ℹ️ About")

    st.write(
        "This chatbot retrieves the most relevant "
        "answer from the NIU FAQ dataset using "
        "NLTK preprocessing, TF-IDF vectorization "
        "and cosine similarity."
    )

    st.write("**Technologies:**")

    st.write(
        "• Python\n"
        "• Pandas\n"
        "• NLTK\n"
        "• Scikit-learn\n"
        "• TF-IDF\n"
        "• Cosine Similarity\n"
        "• Streamlit"
    )

    st.write(
        f"**Total FAQs:** {len(faq_df)}"
    )

    st.write(
        f"**Match threshold:** "
        f"{SIMILARITY_THRESHOLD:.2f}"
    )

    st.write(
        f"**Categories:** "
        f"{faq_df['category'].nunique()}"
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
