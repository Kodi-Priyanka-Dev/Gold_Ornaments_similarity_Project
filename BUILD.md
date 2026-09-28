# Build, Installation, and Deployment Guide

This document provides detailed instructions for setting up, building, and deploying the Gold Ornament Similarity Search application.

## 1. Local Installation and Setup

### Prerequisites
- **Python**: Version 3.9 or higher.
- **Git**: For version control.
- **Virtual Environment Tool**: `venv` or `conda`.
- **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA for faster inference (requires compatible PyTorch build).

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Gold searching"
   ```

2. **Create and Activate a Virtual Environment**
   It's highly recommended to isolate dependencies.
   - Using `venv` (Windows):
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - Using `venv` (macOS/Linux):
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**
   Install the required Python packages from the root directory:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *Note: For GPU support, you might need to install PyTorch manually according to your CUDA version from the [official PyTorch website](https://pytorch.org/get-started/locally/).*

4. **Verify Data Assets**
   Ensure the following directories are present and populated correctly in the root folder:
   - `dataset/`: Contains the reference images of gold ornaments.
   - `models/`: Contains the pre-trained model weights (e.g., `best_model.pth`).
   - `Embeddings/`: Contains pre-computed features (`.npy` and `pkl` files).

## 2. Running Locally for Development

1. **Start the Flask Backend Server**
   Navigate to the backend directory and start the application:
   ```bash
   cd backend
   python app.py
   ```
   Wait for the console output indicating that the ML Service is ready.

2. **Access the Application**
   - Local: Open your web browser and go to `http://localhost:5000`
   - Network: Open a browser on another device on the same network and go to `http://<YOUR_COMPUTER_IP>:5000`

## 3. Deployment Guide

To deploy this application in a production environment, use a production-grade WSGI server (like Gunicorn or uWSGI) instead of the built-in Flask development server, optionally alongside a reverse proxy (like Nginx).

### Prerequisites for Production
- A Linux server (e.g., Ubuntu) or a cloud platform VM.
- Python 3.9+ installed on the server.

### Example: Deploying with Gunicorn and Nginx on Ubuntu

1. **Install Gunicorn and Nginx**
   ```bash
   sudo apt update
   sudo apt install nginx
   pip install gunicorn
   ```

2. **Run with Gunicorn**
   Inside the `backend` directory, start the app using Gunicorn:
   ```bash
   cd backend
   # Binds to localhost:8000
   # Note: For ML workloads, limit workers based on available RAM.
   gunicorn -w 1 -b 127.0.0.1:8000 app:app
   ```
   *Note: Because the model and embeddings are loaded into memory globally, multiple Gunicorn workers will multiply memory usage. Consider using a single worker or ensuring sufficient RAM (e.g., 4GB-8GB+).*

3. **Configure Nginx as a Reverse Proxy**
   Create a new Nginx server block:
   ```bash
   sudo nano /etc/nginx/sites-available/gold-search
   ```
   Add the following configuration:
   ```nginx
   server {
       listen 80;
       server_name your_domain_or_ip;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           
           # Increase upload size limit for image uploads
           client_max_body_size 10M; 
       }
   }
   ```

4. **Enable the Site and Restart Nginx**
   ```bash
   sudo ln -s /etc/nginx/sites-available/gold-search /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

5. **Set up a Systemd Service (Optional but Recommended)**
   To ensure Gunicorn starts on boot, create a systemd service file:
   ```bash
   sudo nano /etc/systemd/system/gold-search.service
   ```
   ```ini
   [Unit]
   Description=Gunicorn instance to serve Gold Search
   After=network.target

   [Service]
   User=your_username
   Group=www-data
   WorkingDirectory=/path/to/Gold searching/backend
   Environment="PATH=/path/to/Gold searching/.venv/bin"
   ExecStart=/path/to/Gold searching/.venv/bin/gunicorn -w 1 -b 127.0.0.1:8000 app:app

   [Install]
   WantedBy=multi-user.target
   ```
   Enable and start the service:
   ```bash
   sudo systemctl start gold-search
   sudo systemctl enable gold-search
   ```

## 4. Troubleshooting

- **Out of Memory (OOM) Errors:** The model and embeddings require significant RAM. If the server crashes on startup, check your available RAM. If deploying with multiple Gunicorn workers, reduce the worker count (`-w 1`).
- **File Upload Limits:** Ensure your web server (Nginx/Apache) allows file uploads up to the size of your query images (e.g., `client_max_body_size` in Nginx).
- **Missing Embeddings/Models:** Ensure `models/best_model.pth` and `Embeddings/` are correctly copied to the deployment server.
