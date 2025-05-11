# 🛸 Introduction:

An face-searched application based on FastAPI, with supports variety types of features. 

<br />


# 🔗 Installing:
1. Clone this project:
```
git clone -b baseline https://github.com/nlp4everyone/face_ekyc
```
2. Go inside project:
```
cd face_ekyc
```
3. Create .env file from .env.sample, then change env variables if necessary:
```
cp .env.sample .env
```
4. Build the project, service port will be opened as FASTAPI_PORT defined in .env file:
```
bash build_docker.sh
```
5. Testing with locust:
```
locust -f testing/measure_load.py --host=http://0.0.0.0:8990
```

# 📃 Intergrations:
- 📂 Containerization: Docker
- 🖥️ Web Framework: FastAPI
- 🗃️ Vector Store: Qdrant
- 🐥 RPS (Request Per Second): Currently, with my baseline setup, with NVIDIA 3060, this system reach approximately 33RPS ( Consumes nearly 6GB VRAM)
- 🎮 Model: Face Detection 🤫 (Mtcnn), Face Embedding 🤗 (ArcFace, AdaFace)
<br />

# 📔 Related Reference:
- 📖 ArcFace: https://arxiv.org/pdf/1801.07698
- 📖 AdaFace: https://arxiv.org/pdf/2204.00964
<br />

# ⭐ Long-term features:
- 🌀 For commerce purpose: Providing APIs as API-token service, with expiration duration and revokcation plus long-term persistence (Postgres). Also, setup Rate-Limiter to restrict requests at time. 
- ⏰ For authentication APIs: use OAuth2PasswordBearer