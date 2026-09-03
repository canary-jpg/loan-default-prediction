# Lessons Learned — Loan Default Prediction Project

## LinkedIn post (ready to copy/paste)

I just wrapped up a loan default prediction project (data cleaning →
feature engineering → model comparison → SHAP explainability → deployed
API + interactive demo), and the most valuable part wasn't the modeling —
it was three deployment bugs that taught me more about real ML engineering
than any notebook could.

**Bug 1: The model wouldn't load in the container.**
`AttributeError: Can't get attribute '_RemainderColsList'`. Turned out my
Docker image installed the latest scikit-learn, but my model was pickled
with an older version. A model artifact isn't just "the model" — it's
tied to the exact library versions that created it. Fix: pin your
dependencies to match your training environment, not just "whatever's
current."

**Bug 2: Same story, different library.**
XGBoost threw a version warning during unpickling too. Same root cause,
same fix — pin it. Two strikes taught me to check *every* ML library in
requirements.txt for version pinning before deploying, not just the one
that broke first.

**Bug 3: The build kept failing with no clear error — until I noticed it
was downloading a 342MB CUDA library I never asked for.**
My requirements.txt had `jupyter` in it — necessary for local notebook
work, completely unnecessary for a running API. It dragged in a massive
transitive dependency tree that blew past my deploy platform's build
limits. Lesson: what you need to *develop* a project and what you need to
*run* it are different lists. I split them into requirements.txt (full,
local dev) and requirements-app.txt (lean, deployment-only).

None of this showed up in a single notebook cell. All of it showed up the
moment I tried to actually ship something. If you're building a portfolio
project, my honest advice: don't stop at the notebook. Deploy it. The bugs
you'll hit are the ones that actually matter.

🔗 [link to repo] | [link to live demo]

#MachineLearning #DataScience #MLOps #PortfolioProject

---

## Interview talking points

Use these if asked "tell me about a time something broke" or "walk me
through a technical challenge you solved."

### Talking point 1 — Dependency/version pinning (headline story)

**Situation:** After training an XGBoost model locally and building a
FastAPI service around it, I containerized it with Docker for deployment.

**Task:** Get the exact same model, trained and validated locally, running
correctly inside the container.

**Action:** The container crashed on startup trying to load the pickled
model — `AttributeError: Can't get attribute '_RemainderColsList'`. I
traced it to my `requirements.txt` not pinning scikit-learn, so the
container installed a newer version than what I'd trained with, and
newer scikit-learn's internal pickling format wasn't backward-compatible.
I pinned the exact version from my local environment, rebuilt, and hit
the identical problem with XGBoost a moment later — so I checked and
pinned every ML library in the file, not just the one that had already
failed.

**Result:** Working, reproducible container. More importantly, I now
treat "pin your training environment's exact library versions before
deploying" as a standing rule, not something I discover after a crash.

**Why this is a good interview answer:** it shows you understand that
model artifacts are coupled to their environment, not just "the model
file" — a distinction a lot of people learn the hard way once they start
deploying real models.

### Talking point 2 — Build/dependency hygiene

**Situation:** A second deployment (an interactive Streamlit demo of the
same model) failed with an unclear build error after downloading a
suspiciously large (342MB) package.

**Task:** Figure out why a deployment platform's free-tier build was
failing without a clear Python traceback to point to.

**Action:** Read the raw build log line by line rather than assuming it
was another code bug, and noticed `jupyter` in my requirements file was
pulling in a huge, irrelevant dependency tree (Jupyter's full stack, plus
transitive packages meant for GPU environments). I split my dependencies
into a full local-dev file and a lean, deployment-only file containing
just what the running service actually imports.

**Result:** Both services deployed cleanly on the free tier. I now
default to separating dev and runtime dependencies on any project headed
for deployment.

**Why this is a good interview answer:** shows debugging discipline
(reading logs instead of guessing) and an engineering instinct — "what
does this service actually need to run" — that's easy to skip when
you're used to working in notebooks.

### Talking point 3 — Working around a platform bug

**Situation:** A separate deployment attempt (Streamlit Community Cloud)
kept stalling and restarting with no clear cause.

**Task:** Determine whether this was my bug or the platform's, without
burning unlimited time on trial and error.

**Action:** Searched for the exact symptom rather than continuing to
guess-and-check, and found several other developers reporting the same
issue — a known, unresolved bug where that platform ignores explicit
Python version configuration. Rather than waiting on a third-party fix, I
redirected the deployment to a platform I already had working (via
Docker, where I control the Python version directly).

**Result:** Live demo, on schedule, without depending on someone else's
bug fix timeline.

**Why this is a good interview answer:** shows judgment about when to
keep debugging your own code versus recognizing an external constraint
and adapting the plan — a genuinely underrated skill.
