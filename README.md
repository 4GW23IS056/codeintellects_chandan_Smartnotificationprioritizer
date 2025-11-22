# Notify It

**Notify It** is a modern, local hybrid AI-powered notification prioritization system with a colorful glass UI. It allows users to visualize, customize, and prioritize app notifications based on historical behavioral logs or manual preferences.

---

## Features

* **AI-Powered Prioritization**:
  Uses a hybrid model combining a logistic regression trained on notification events and heuristic scoring to rank domains by importance.

* **Manual Reordering**:
  Users can manually adjust the priority of enabled domains with a simple drag-up/down interface.

* **Customizable Notifications**:
  Each domain allows individual settings for:

  * Priority level (`high`, `medium`, `low`)
  * Notification sound (`chime`, `ping`, `bloop`, `melody`)
  * Vibration pattern (`none`, `short`, `long`, `pulse`)
  * Volume (0–100)

* **Reset Functionality**:
  Clear all settings and trained AI models while preserving the sample dataset.

* **Modern UI/UX**:

  * Glassmorphic design with soft blur and gradient effects
  * Animated tiles and buttons for a lively interface
  * Responsive design for desktop and mobile

---

## Tech Stack

* **Frontend**: HTML, CSS (glass UI, animations), Vanilla JavaScript
* **Backend**: Python, Flask
* **Database**: SQLite (`storage.db`) for storing per-domain settings
* **AI Model**: Pure Python logistic regression trained on `sample_dataset.json`
* **API Calls**:

  * `/api/train` – Train the model from the dataset
  * `/api/predict` – Predict domain priority order
  * `/api/settings/<domain>` – Save/load settings
  * `/api/reset` – Reset settings and remove trained model

---

## Project Structure

```
backend/
 ├─ app.py             # Flask backend
 ├─ model.py           # AI model logic (training, prediction)
 ├─ storage.db         # SQLite database (auto-created)
 ├─ sample_dataset.json # Example dataset
 └─ priority_model.json # Trained model weights

frontend/
 ├─ index.html         # Main dashboard
 ├─ page_domain.html   # Domain-specific settings
 └─ assets/
     └─ style.css      # Glass UI + animations

requirements.txt       # Python dependencies
```

---

## Installation

1. **Clone the repository**:

```bash
git clone <repo-url>
cd smart-notif-prioritizer/backend
```

2. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the backend**:

```bash
python app.py
```

5. **Open the frontend**:

Navigate to `http://127.0.0.1:5000/` in your browser.

---

## Usage

1. **Dashboard**:

   * Click a domain tile to visit its settings page.
   * Right-click a tile to enable/disable it.

2. **AI Prioritize**:

   * Click **AI Prioritize** to train the model on `sample_dataset.json` and reorder domains automatically.

3. **Manual Mode**:

   * Click **Manual Mode** to reorder domains manually.
   * Use ▲/▼ buttons and click **Apply Order**.

4. **Reset**:

   * Click **Reset** to clear all settings and remove the trained model.

---

## Dataset

The project comes with a sample dataset (`sample_dataset.json`) containing per-notification logs:

```json
{
  "events": [
    {"domain":"Messengers","received":1706092000,"opened":1,"dismissed":0,"action_clicked":1,"delay_seconds":3},
    ...
  ],
  "meta": {"description":"Per-notification event logs (opened/dismissed/action/delay) — used for learning priorities"}
}
```

The AI model uses these logs to learn which notifications are most likely important to the user.

---

## Notes

* The AI uses a **hybrid scoring** system:

  * `heuristic score`: Based on historical interaction rate & recency
  * `model score`: Logistic regression on per-event features
  * `blend_alpha` determines weight between model and heuristic.

* The frontend communicates with the backend via RESTful API endpoints.

* SQLite database (`storage.db`) stores per-domain settings persistently.

