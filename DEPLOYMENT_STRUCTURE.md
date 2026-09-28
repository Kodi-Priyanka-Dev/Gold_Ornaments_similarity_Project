# 🚀 Deployment Folder Structure

When you deploy this application to a production server (like AWS, Azure, Heroku, or a local network server), you do **not** need the entire training dataset or all the training scripts (like `main.py`).

Here is the exact, minimized folder structure you need for deployment (which is now completely handled by the Docker container):

```text
Gold searching/
├── backend/
│   ├── api/
│   │   └── routes.py             # API endpoints
│   ├── services/
│   │   ├── similarity_search.py  # ML logic for similarity and classification
│   │   ├── llm_helper.py         # Gemini/OpenRouter API integration
│   │   └── jewelry_validator.py  # CLIP-based jewelry check
│   ├── app.py                    # The Flask server entry point
│   ├── gunicorn_conf.py          # Production WSGI server configuration
│   └── uploads/                  # Temporary storage (created automatically)
│
├── frontend/                     
│   ├── index.html                # The web interface
│   ├── style.css                 # Web styling
│   └── script.js                 # Frontend API logic
│
├── Dockerfile                    # The multi-stage build instructions for production
├── docker-compose.yaml           # Local / production compose orchestration
├── requirements.txt              # Dev dependencies
├── requirements-serve.txt        # Minimized production runtime dependencies
└── metadata.json                 # Core inventory metadata used for search results
```

## Deployment Notes:
1. **Models:** You no longer need to ship large `best_model.pth` files. The ML models (ConvNeXt and CLIP-ViT) are automatically baked into the Docker image during the build process to guarantee fast cold starts and prevent offline deployment failures.
2. **Qdrant & Embeddings:** All high-dimensional search operations rely on your cloud Qdrant cluster. You no longer need to carry around massive `.npy` or `.pkl` files in production!
3. **Ignored Files:** The `main.py` training code, `server.log`, and any quick one-off migration scripts are purely for local development and are explicitly excluded from the Docker build context.

---

## 🚀 Docker Deployment (Recommended)

To successfully deploy and run the project using Docker, follow these steps:

### 1. Build the Docker Image
Docker will automatically install the minimized `requirements-serve.txt` and bake the ML models into the image layers.
```bash
docker build -t gold-search-app:latest .
```

### 2. Run the Container
You must provide the Qdrant credentials as environment variables to the container.
```bash
docker run -d \
  --name gold-search-api \
  -p 5000:5000 \
  -e QDRANT_URL="your_qdrant_cluster_url" \
  -e QDRANT_API_KEY="your_qdrant_api_key" \
  -e ALLOWED_ORIGINS="https://your-frontend-domain.com" \
  gold-search-app:latest
```

---

## ☁️ Deploying to Azure App Service (Docker / ACR)

Since you are pushing images to Azure Container Registry (`sithafal.azurecr.io`), the recommended path is **Azure App Service for Containers**.

### 1. Push your Image to ACR
Log in to your Azure Container Registry and push your tagged image:
```bash
az acr login --name sithafal
docker push sithafal.azurecr.io/gold-search-app:v4
```

### 2. Create the App Service from the Container
Use the Azure CLI to create a web app directly from your container registry image:
```bash
az webapp create \
  --resource-group your-resource-group \
  --plan your-app-service-plan \
  --name your-unique-app-name \
  --deployment-container-image-name sithafal.azurecr.io/gold-search-app:v4
```

### 3. Configure Environment Variables
Your application requires the Qdrant API keys and a secure CORS configuration to function. Add these to Azure's App Settings:
```bash
az webapp config appsettings set \
  --name your-unique-app-name \
  --settings \
  QDRANT_URL="your_qdrant_cluster_url" \
  QDRANT_API_KEY="your_qdrant_api_key" \
  ALLOWED_ORIGINS="https://your-unique-app-name.azurewebsites.net"
```

### 4. Enable Continuous Deployment (Optional)
You can configure App Service to pull the latest image automatically via webhooks whenever you run a new `docker push`.
```bash
az webapp deployment container config --enable-cd true --name your-unique-app-name
```

Your containerized API will now be live and publicly accessible!

---

## 📦 Jewelry Images: Azure Blob Storage

`Jewellery_Data/` is no longer carried on local disk or baked into the Docker
image - every image lives in Blob Storage instead, and `backend/services/blob_storage.py`
serves them through the existing `/api/images/<path>` route (falls back to
Blob only when a file isn't found locally, so nothing else changes). Right
now that data's only copy is in the local Azurite container started by
`docker-compose.yaml` - use these steps to get it into a real Azure Storage
account.

### 1. Create the Storage Account and Container
```bash
az storage account create \
  --name <youruniqueaccountname> \
  --resource-group your-resource-group \
  --location <region> \
  --sku Standard_LRS \
  --kind StorageV2

az storage container create \
  --account-name <youruniqueaccountname> \
  --name jewellery-data \
  --auth-mode login
```

### 2. Grant Access - Managed Identity (no keys, ever)
The App Service (or whatever compute runs the API) reads Blob Storage with
its own managed identity, not a shared key:
```bash
az webapp identity assign --name your-unique-app-name --resource-group your-resource-group
# note the principalId this prints

az role assignment create \
  --assignee <principalId> \
  --role "Storage Blob Data Contributor" \
  --scope $(az storage account show --name <youruniqueaccountname> --resource-group your-resource-group --query id -o tsv)
```
Also grant yourself that same role (`--assignee <your-az-login-user>`) so
the one-time migration below can write to the account from your machine.

### 3. Point the App at the Real Account
```bash
az webapp config appsettings set \
  --name your-unique-app-name \
  --resource-group your-resource-group \
  --settings \
  AZURE_STORAGE_ACCOUNT_URL="https://<youruniqueaccountname>.blob.core.windows.net" \
  AZURE_STORAGE_CONTAINER="jewellery-data"
```
Do **not** set `AZURE_STORAGE_CONNECTION_STRING` in production - its presence
would bypass managed identity. It's local-dev/Azurite only (see `.env.example`).

### 4. Migrate the Data (Azurite -> Real Azure, one-time)
```bash
az login   # so DefaultAzureCredential can act as you
export AZURE_STORAGE_DEST_ACCOUNT_URL="https://<youruniqueaccountname>.blob.core.windows.net"
python backend/scripts/migrate_to_real_azure.py
```
This copies every blob straight from the local Azurite container (source of
truth, since `Jewellery_Data/` was removed from disk) into the real account -
no need to have the original files anywhere. `backend/scripts/backfill_blob_storage.py`
remains available too, for the (rarer) case of seeding Blob Storage straight
from a local image folder.
