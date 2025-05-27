# TranslationAPI

## Backend setup

```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

Access backend: `http://127.0.0.1:8000/docs`

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Access frontend: `http://localhost:5173/`

## Docker setup

Select whether to use the CPU or GPU (requires a CUDA compatible GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) set up):

```bash
docker compose --profile cpu up
docker compose --profile gpu up
```

Launch app: `http://localhost:8080/`
