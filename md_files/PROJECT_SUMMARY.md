# Gold Ornament Similarity Search: End-to-End Project Breakdown

This document serves as a comprehensive technical explanation of the entire project, detailing the architecture, parameters, and the role of every file in the system.

---

## 1. The Core Machine Learning Model & Training

### Model Selection
- **Architecture:** We used **ConvNeXt Base**, a state-of-the-art Convolutional Neural Network (CNN) that competes directly with Vision Transformers (ViTs) in terms of accuracy but retains the speed and simplicity of standard CNNs.
- **Library:** Loaded via `timm` (PyTorch Image Models).
- **Number of Classes:** The model was originally configured for `NUM_CLASSES = 6` (representing the 6 categories of jewelry in your dataset, such as Bracelets, Necklaces, Earrings, etc.).

### Image Processing & Parameters
Before any image can be fed into the model (during both training and searching), it undergoes a strict transformation pipeline:
- **Resolution (Resizing):** Every image is resized to exactly `224 x 224` pixels. This is the standard input size for ConvNeXt models.
- **Normalization:** The pixels are normalized using the ImageNet standard means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. This ensures the model processes colors and lighting consistently.

### The "Training" & Feature Extraction Phase
While the actual training loop is completed, the most critical part of this search engine is **Feature Extraction**.
- Instead of using the model to *classify* an image (e.g., guessing "This is a necklace"), we removed the final classification layer (`model.reset_classifier(0)`).
- This turns the model into a feature extractor. When an image passes through, the model outputs a dense mathematical vector (an "embedding") that represents the visual characteristics of that specific piece of jewelry.
- We ran this extraction across all **8,252 images** in your dataset and saved the resulting vectors.

---

## 2. File-by-File Breakdown

### The Original ML Scripts (`main.py/`)
- **`train.py` & `dataset.py`:** These scripts were originally used to load the dataset and train the `convnext_base` model to recognize the 6 categories of jewelry, saving the best weights to `models/best_model.pth`.
- **`feature_extraction.py`:** This script ran every single image in your dataset through the trained model, converting them into mathematical vectors, and saving them to `Embeddings/image_features.npy` and `image_paths.pkl`.
- **`similarity_search.py`:** The original CLI script that proved the concept worked by taking a query image, turning it into a vector, and comparing it against the database.

### The Backend (`backend/`)
- **`app.py`:** The main entry point for the Flask web server. It configures `Flask-CORS` so the frontend is allowed to talk to it, binds to `0.0.0.0` (allowing local network access), and listens on Port 5000.
- **`services/similarity_search.py`:** The heart of the backend. When the server starts, this file loads the `best_model.pth` and the 8,252 vectors into RAM. It exposes a `find_similar()` function that mathematically compares a newly uploaded image against the 8,252 vectors using **Cosine Similarity** (a distance metric where 1.0 means an exact match).
- **`api/routes.py`:** This acts as the bridge between the web and the python logic. 
  - The `/search` route accepts the image uploaded from the browser, saves it temporarily, and sends it to `find_similar()`.
  - The `/images/<path>` route securely serves the raw `.jpg` files from your hard drive so the web browser can display the results.

### The Frontend (`frontend/`)
- **`index.html`:** Defines the structure of the web page (the upload box, buttons, and the grid where results appear).
- **`style.css`:** Contains all the visual design rules. It uses modern "glassmorphism" (blurring background elements), dark-mode styling (`#0d0f14`), and dynamic hover animations to make the site feel premium.
- **`script.js`:** The interactive logic. It handles the drag-and-drop mechanics, takes the image file, and sends a `fetch` request (HTTP POST) to the backend's `/search` route. When the backend replies with the top matches, this script dynamically draws them onto the screen. It also separates results into "Exact Matches" if their similarity score is `≥97%`.

---

## 3. End-to-End Workflow (What happens when you click "Search")

1. **User Action:** You drag an image into the browser. `script.js` instantly creates a local preview so you can see what you uploaded.
2. **The Request:** When you click Search, `script.js` bundles the image into a `FormData` object and sends it over HTTP to `http://127.0.0.1:5000/api/search`.
3. **Backend Processing:** `routes.py` receives the image, saves it to the `backend/uploads/` folder, and passes the file path to `services/similarity_search.py`.
4. **ML Inference:** The service opens the image, resizes it to `224x224`, normalizes the colors, and pushes it through the `convnext_base` neural network.
5. **Similarity Calculation:** The network outputs a feature vector. The system calculates the dot product (Cosine Similarity) between this new vector and all 8,252 vectors currently sitting in RAM.
6. **Ranking:** The scores are sorted from highest to lowest. The top 20 paths and percentage scores are formatted into a JSON response and sent back to the frontend.
7. **Rendering:** `script.js` receives the JSON. It uses the `/api/images/` route to ask the backend for the actual `.jpg` files and draws them on your screen, grouped by how perfectly they match!
