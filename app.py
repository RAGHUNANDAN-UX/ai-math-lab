import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI Math Lab", page_icon="🧠", layout="wide")

# ───────────────────────── Styling ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {font-family:'Inter',sans-serif; color:#14110f;}
.stApp {background:radial-gradient(900px 600px at 90% -5%, #ffe2cd, transparent 60%), #f7f1ec;}
[data-testid="stToolbar"], footer, #MainMenu {display:none;}
header {background:transparent;}
.block-container {padding-top:2rem; max-width:1200px;}
@keyframes up {from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:none}}
.clay {border-radius:24px;padding:20px 24px;margin:7px 0;background:#fff;border:1px solid #f0e6dc;
 box-shadow:0 10px 30px rgba(120,80,40,.08);animation:up .7s both;transition:transform .4s, box-shadow .4s, border-color .4s;}
.clay:hover {transform:translateY(-6px);border-color:#ff6a1a;box-shadow:0 22px 44px rgba(255,106,26,.2);}
.c0{background:linear-gradient(145deg,#fff6e0,#fff)} .c1{background:linear-gradient(145deg,#e9f5ff,#fff)}
.c2{background:linear-gradient(145deg,#ffe9e2,#fff)} .c3{background:linear-gradient(145deg,#fff0e4,#fff)}
.c4{background:linear-gradient(145deg,#e6f8ec,#fff)}
.eyebrow {color:#8a7f76;font-size:.75rem;letter-spacing:.22em;text-transform:uppercase;}
.lbl {color:#7a6f66;font-size:.92rem;} .name {font-size:1.05rem;font-weight:700;}
h1.hero {font-size:3.6rem;font-weight:800;line-height:1.05;margin:12px 0;letter-spacing:-.03em;}
h1.hero span {color:#ff6a1a;}
.stat {font-size:2rem;font-weight:800;line-height:1.1;}
.brand {font-size:1.6rem;font-weight:800;margin:6px 0 20px 6px;letter-spacing:-.03em;} .brand span {color:#ff6a1a;}
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#ffffff 0%,#fff1e6 100%);border-right:1px solid #f0e6dc;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {width:100%;padding:10px 16px;border-radius:999px;transition:background .3s, transform .3s;cursor:pointer;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:#ffe3cf;transform:translateX(6px);}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {background:#14110f;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {color:#fff;font-weight:700;}
.stButton>button {border:0;border-radius:999px;padding:.5rem 1.5rem;color:#fff;font-weight:600;background:#14110f;}
.stButton>button:hover {color:#fff;background:#ff6a1a;}
</style>
""", unsafe_allow_html=True)

ORANGE, DARK, GREEN, BLUE = "#ff6a1a", "#14110f", "#22c55e", "#4c9cf0"


def card(html, cls="white", delay=0):
    st.markdown(f'<div class="clay {cls}" style="animation-delay:{delay}s">{html}</div>', unsafe_allow_html=True)


# ───────────────────────── 1. Digit recognition (neural network) ─────────────────────────
@st.cache_resource(show_spinner="Training 3 AI models on 1,347 handwritten digits...")
def train_digits():
    d = load_digits()
    Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25, random_state=1, stratify=d.target)
    models = {
        "k-Nearest Neighbours": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=1),
    }
    acc = {}
    for n, m in models.items():
        m.fit(Xtr, ytr)
        acc[n] = float((m.predict(Xte) == yte).mean())
    return models, acc, Xte, yte


def digits_page():
    st.markdown("## 🔢 Handwritten Digit Recognizer")
    st.caption("Three different AI models learn to read 8×8 pixel handwritten digits. Add noise and see who survives.")
    models, acc, Xte, yte = train_digits()

    cols = st.columns(3)
    for c, (n, a) in zip(cols, acc.items()):
        with c:
            card(f'<div class="stat">{a:.1%}</div><div class="lbl">{n}<br>test accuracy</div>', "c4")

    l, r = st.columns([1, 1.3])
    with l:
        idx = st.slider("Pick a test image", 0, len(Xte) - 1, 12)
        noise = st.slider("Add noise (damage the image)", 0.0, 8.0, 0.0, 0.5)
        rng = np.random.default_rng(idx)
        x = np.clip(Xte[idx] + rng.normal(0, noise, 64), 0, 16)
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(x.reshape(8, 8), cmap="Oranges")
        ax.axis("off")
        ax.set_title(f"Real digit: {yte[idx]}", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)
    with r:
        rows, probs = [], {}
        for n, m in models.items():
            p = m.predict_proba(x.reshape(1, -1))[0]
            pred = int(np.argmax(p))
            probs[n] = p
            rows.append({"Model": n, "Prediction": pred, "Confidence": f"{p[pred]:.0%}",
                         "Result": "✅ correct" if pred == yte[idx] else "❌ wrong"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pick = st.selectbox("See probabilities of", list(models), index=2)
        st.bar_chart(pd.DataFrame({"probability": probs[pick]}, index=[str(i) for i in range(10)]), color=ORANGE)

    with st.expander("📐 Math used here"):
        st.markdown("""
- **Image = vector:** each 8×8 image is a vector `x ∈ ℝ⁶⁴` (pixel brightness 0 to 16).
- **k-NN:** predicts the majority label of the 3 closest training images, using Euclidean distance `d = √Σ(xᵢ − yᵢ)²`.
- **Logistic regression:** `P(class k) = e^(wₖ·x) / Σⱼ e^(wⱼ·x)` (softmax), trained by minimising cross-entropy.
- **Neural network:** `h = ReLU(W₁x + b₁)`, `output = softmax(W₂h + b₂)`. Weights are learned with gradient descent (partial derivatives and the chain rule = backpropagation).
- **Noise:** `x' = clip(x + N(0, σ²), 0, 16)` (Gaussian random noise).
""")


# ───────────────────────── 2. Sentiment AI (Naive Bayes) ─────────────────────────
BASE_DATA = [
    ("the lecture was amazing and very clear", 1), ("i love this college and the friendly teachers", 1),
    ("great marks today so happy", 1), ("the lab session was fun and useful", 1),
    ("excellent project and the team worked well", 1), ("i enjoyed the fest it was awesome", 1),
    ("good food in the canteen today", 1), ("the faculty explained everything nicely", 1),
    ("wonderful day with my friends", 1), ("i am proud of my result", 1),
    ("our group won the ideathon what a great feeling", 1), ("the new library is beautiful and peaceful", 1),
    ("easy exam and i did well", 1), ("super helpful notes thank you", 1),
    ("nice and interesting class", 1), ("best day ever", 1),
    ("the lecture was boring and too long", 0), ("i hate the traffic and i was late", 0),
    ("bad marks i am very sad", 0), ("the exam was too hard and unfair", 0),
    ("worst canteen food ever", 0), ("the wifi is slow and useless", 0),
    ("i failed the test so disappointed", 0), ("terrible teaching nothing was clear", 0),
    ("too much homework i am tired and stressed", 0), ("the lab was a waste of time", 0),
    ("my project crashed and i am angry", 0), ("boring class and sleepy students", 0),
    ("horrible day nothing went right", 0), ("poor internet and bad service", 0),
    ("i am worried about the exam", 0), ("awful noise in the hostel", 0),
]


def train_nb(data):
    texts, labels = zip(*data)
    vec = CountVectorizer()
    X = vec.fit_transform(texts)
    nb = MultinomialNB(alpha=1.0).fit(X, labels)
    return vec, nb


def sentiment_page():
    st.markdown("## 💬 Sentiment AI (Naive Bayes)")
    st.caption("Type a student-life sentence. The AI says Positive or Negative and shows which words influenced it. You can also teach it new sentences.")
    ss = st.session_state
    ss.setdefault("extra", [])
    data = BASE_DATA + ss.extra
    vec, nb = train_nb(data)

    text = st.text_input("Your sentence", "the teacher explained very well and the class was fun")
    if text.strip():
        p = nb.predict_proba(vec.transform([text]))[0]
        label = "😊 Positive" if p[1] >= p[0] else "😞 Negative"
        a, b = st.columns(2)
        with a:
            card(f'<div class="stat">{label}</div><div class="lbl">Positive {p[1]:.0%} · Negative {p[0]:.0%}</div>', "c4" if p[1] >= p[0] else "c2")
        known = [w for w in vec.build_analyzer()(text) if w in vec.vocabulary_]
        with b:
            card(f'<div class="stat">{len(known)}</div><div class="lbl">words the AI knows from {len(data)} training sentences</div>', "c1", .1)
        if known:
            rows = []
            for w in dict.fromkeys(known):
                j = vec.vocabulary_[w]
                score = float(nb.feature_log_prob_[1][j] - nb.feature_log_prob_[0][j])
                rows.append({"Word": w, "Score (log-odds)": round(score, 2), "Pushes towards": "Positive" if score > 0 else "Negative"})
            st.dataframe(pd.DataFrame(rows).sort_values("Score (log-odds)"), hide_index=True, width="stretch")
        else:
            st.info("The AI does not know any of these words yet. Teach it below.")

    with st.expander("🧑‍🏫 Teach the AI a new sentence"):
        ns = st.text_input("New sentence", key="ns")
        nl = st.radio("It is", ["Positive", "Negative"], horizontal=True)
        if st.button("Add and retrain") and ns.strip():
            ss.extra.append((ns.strip().lower(), 1 if nl == "Positive" else 0))
            st.rerun()
        if ss.extra:
            st.caption(f"You added {len(ss.extra)} sentence(s).")
            if st.button("Reset my sentences"):
                ss.extra = []
                st.rerun()

    with st.expander("📐 Math used here"):
        st.markdown("""
- **Bag of words:** each sentence becomes a vector of word counts.
- **Bayes' theorem:** `P(class | words) ∝ P(class) · Π P(wordᵢ | class)`.
- **Laplace smoothing:** `P(w | class) = (count(w, class) + 1) / (total words in class + V)` so unseen words never give zero.
- **Log-odds score** of a word = `ln P(w | positive) − ln P(w | negative)`. Positive means it pushes the answer to Positive.
- The AI is just counting and multiplying probabilities. No magic.
""")


# ───────────────────────── 3. Train an AI live (gradient descent) ─────────────────────────
@st.cache_data
def marks_data():
    rng = np.random.default_rng(7)
    hrs = rng.uniform(0.5, 9.5, 40)
    marks = np.clip(32 + 6.2 * hrs + rng.normal(0, 5, 40), 0, 100)
    return hrs, marks


def gradient_descent(x, y, lr, epochs):
    w = b = 0.0
    hist = [(w, b, float(np.mean((w * x + b - y) ** 2)))]
    for _ in range(epochs):
        err = w * x + b - y
        gw, gb = 2 * np.mean(err * x), 2 * np.mean(err)   # partial derivatives of the loss
        w, b = w - lr * gw, b - lr * gb
        loss = float(np.mean((w * x + b - y) ** 2))
        hist.append((w, b, loss))
        if not np.isfinite(loss) or loss > 1e6:
            break
    return hist


def learn_page():
    st.markdown("## 📈 Train an AI Live: Marks Predictor")
    st.caption("Watch an AI learn the line `marks = w × hours + b` step by step using gradient descent.")
    hrs, marks = marks_data()
    x, y = hrs / 10, marks / 100                      # scale to 0..1 so training is stable
    c1, c2 = st.columns(2)
    lr = c1.slider("Learning rate", 0.01, 2.2, 0.3, 0.01)
    epochs = c2.slider("Training steps (epochs)", 1, 500, 150)
    hist = gradient_descent(x, y, lr, epochs)
    diverged = (not np.isfinite(hist[-1][2])) or hist[-1][2] > 1e6
    step = st.slider("Move through training", 0, len(hist) - 1, len(hist) - 1)
    w, b, loss = hist[step]

    # closed-form answer using linear algebra (normal equation)
    A = np.c_[x, np.ones_like(x)]
    w_best, b_best = np.linalg.solve(A.T @ A, A.T @ y)

    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.scatter(hrs, marks, color=BLUE, s=22, label="students")
        if not diverged or step < 5:
            xs = np.array([0, 10])
            ax.plot(xs, (w * xs / 10 + b) * 100, color=ORANGE, lw=3, label=f"AI line (step {step})")
        ax.plot([0, 10], [(w_best * 0 + b_best) * 100, (w_best + b_best) * 100], "--", color=GREEN, lw=2, label="best possible line")
        ax.set_ylim(0, 105); ax.set_xlabel("hours studied"); ax.set_ylabel("marks")
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with r:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot([h[2] for h in hist], color=DARK, lw=2)
        ax.scatter([step], [hist[step][2]], color=ORANGE, zorder=3)
        ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss (mean squared error)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    if diverged:
        st.error("💥 The learning rate is too big. The AI overshoots and the error explodes. Reduce it.")
    else:
        a, b2, c = st.columns(3)
        with a: card(f'<div class="stat">{loss * 100 * 100:.1f}</div><div class="lbl">loss now (marks²)</div>', "c2")
        with b2: card(f'<div class="stat">{w * 10:.1f}</div><div class="lbl">marks gained per extra hour (learned)</div>', "c3", .1)
        with c: card(f'<div class="stat">{w_best * 10:.1f}</div><div class="lbl">marks per hour (exact answer)</div>', "c4", .2)
        h = st.number_input("Predict my marks if I study (hours)", 0.0, 12.0, 6.0, 0.5)
        st.success(f"The AI predicts **{min(100, (w * h / 10 + b) * 100):.0f} marks** for {h:g} hours of study.")

    with st.expander("📐 Math used here"):
        st.markdown("""
- **Model:** `ŷ = w·x + b`.
- **Loss (mean squared error):** `L = (1/n) Σ (wxᵢ + b − yᵢ)²`.
- **Partial derivatives:** `∂L/∂w = (2/n) Σ (ŷᵢ − yᵢ)xᵢ` and `∂L/∂b = (2/n) Σ (ŷᵢ − yᵢ)`.
- **Gradient descent update:** `w ← w − η·∂L/∂w`, `b ← b − η·∂L/∂b`, where η is the learning rate.
- **Exact solution (linear algebra):** the normal equation `θ = (AᵀA)⁻¹ Aᵀy`, where A has columns `[x, 1]`. This is the green dashed line.
- Too small η learns slowly. Too large η makes the steps overshoot and diverge.
""")


# ───────────────────────── 4. K-Means clustering ─────────────────────────
def kmeans_page():
    st.markdown("## 🎯 K-Means: AI Finds Groups by Itself")
    st.caption("No labels given. The AI groups students by hours studied and hours on the phone.")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Number of groups (k)", 1, 8, 3)
    spread = c2.slider("How spread out are the students", 0.5, 2.5, 1.0, 0.1)
    seed = c3.number_input("Random seed", 0, 99, 4)
    rng = np.random.default_rng(int(seed))
    centres = np.array([[2, 7], [7, 2], [7, 7.5], [2.5, 2.2]])
    pts = np.vstack([c + rng.normal(0, spread * 0.9, (35, 2)) for c in centres[:3]])
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(pts)
    inertia = [KMeans(n_clusters=i, n_init=10, random_state=0).fit(pts).inertia_ for i in range(1, 9)]

    l, r = st.columns(2)
    with l:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pts[:, 0], pts[:, 1], c=km.labels_, cmap="tab10", s=26)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="black", marker="X", s=180, edgecolor="white", label="centroids")
        ax.set_xlabel("hours studied per day"); ax.set_ylabel("hours on phone per day")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with r:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(range(1, 9), inertia, marker="o", color=DARK)
        ax.scatter([k], [inertia[k - 1]], color=ORANGE, s=120, zorder=3)
        ax.set_xlabel("k"); ax.set_ylabel("total squared distance (inertia)"); ax.set_title("Elbow method")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    st.info("Look for the 'elbow' in the right graph: the k after which the line stops dropping fast. This data was made from 3 real groups.")
    with st.expander("📐 Math used here"):
        st.markdown("""
- **Distance:** `d(p, c) = √((x₁−c₁)² + (y₁−c₂)²)` (Euclidean distance).
- **Step 1:** assign every point to its nearest centroid.
- **Step 2:** move each centroid to the mean of its points: `c = (1/m) Σ pᵢ`.
- Repeat until nothing changes. This minimises `inertia = Σ ‖p − c‖²`.
""")


# ───────────────────────── Home ─────────────────────────
def home():
    card('<div class="name" style="text-align:center;font-size:1.4rem">BGS College of Engineering and Technology</div>', "c3")
    l, r = st.columns([1.2, 1])
    with l:
        st.markdown('<div class="eyebrow" style="margin-top:18px">Python · Machine Learning · Maths</div>'
                    '<h1 class="hero">AI<br><span>Math Lab</span></h1>'
                    '<p class="lbl" style="font-size:1.05rem;max-width:460px">Four mini AIs you can play with: read digits, judge feelings, learn a line live and find hidden groups. Every page shows the maths behind it.</p>'
                    '<div class="name" style="margin-top:14px">Built by RAGHUNANDAN</div>', unsafe_allow_html=True)
    with r:
        st.markdown('<div style="font-size:9rem;text-align:center;filter:drop-shadow(0 24px 24px rgba(120,70,30,.25))">🧠</div>', unsafe_allow_html=True)
    items = [("🔢", "Digit Recognizer", "Neural network, k-NN and logistic regression read handwriting.", "Linear algebra, softmax, chain rule"),
             ("💬", "Sentiment AI", "Naive Bayes judges positive or negative text. You can teach it.", "Probability, Bayes' theorem"),
             ("📈", "Train AI Live", "Watch gradient descent learn a marks predictor.", "Partial derivatives, matrices"),
             ("🎯", "K-Means Groups", "AI discovers groups without labels.", "Distance, averages, optimisation")]
    for i, (c, (ic, t, d, m)) in enumerate(zip(st.columns(4), items)):
        with c:
            card(f'<div style="font-size:2rem">{ic}</div><div class="name">{t}</div><div class="lbl">{d}</div><div class="lbl" style="margin-top:8px"><b>Maths:</b> {m}</div>', f"c{i}", .1 * i)


# ───────────────────────── Navigation ─────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">AI<span>.</span>math lab</div>', unsafe_allow_html=True)
    page = st.radio("nav", ["Home", "Digit Recognizer", "Sentiment AI", "Train AI Live", "K-Means Groups"], label_visibility="collapsed")
    st.markdown('<div class="lbl" style="margin-top:28px">Built by</div><div class="name">RAGHUNANDAN</div>', unsafe_allow_html=True)

{"Home": home, "Digit Recognizer": digits_page, "Sentiment AI": sentiment_page,
 "Train AI Live": learn_page, "K-Means Groups": kmeans_page}[page]()
