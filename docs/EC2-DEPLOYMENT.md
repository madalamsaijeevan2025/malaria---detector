# Deploying to an AWS EC2 Server

Step-by-step guide to putting the malaria-detection service on an EC2
instance, with **every command explained** so you can present it and
defend it in an interview/exam.

```
Your PC                     EC2 instance (AWS cloud)
┌──────────────┐   scp    ┌─────────────────────────────┐
│ malaria-     │─────────▶│ extract → docker compose up │
│ detect/      │          │ FastAPI :8000 + MobileNetV2 │
└──────────────┘          └─────────────────────────────┘
```

## 0. What you need first

| Requirement | Why |
|---|---|
| EC2 instance, **t3.medium** or larger (2 vCPU, 4 GiB RAM) | The container is capped at 2 CPUs / 3 GB; smaller instances OOM during the build |
| Root volume **≥ 20 GB** | The Docker image is ~2.2 GB and the build needs temporary space (default 8 GB can fill up) |
| A **key pair** (`.pem` file) downloaded from AWS | SSH access; AWS has no password login |
| Security group allowing **SSH (22)** from your IP | So you can connect at all |

> On EC2 you are charged by the hour. **Terminate the instance when the
> demo is done** (last section) — otherwise it keeps billing you.

---

## 1. Package the project on your PC

```bash
cd /c/Users/Admin
tar -czf malaria-detect.tar.gz \
  --exclude='malaria-detect/.venv' \
  --exclude='malaria-detect/.cache' \
  --exclude='malaria-detect/__pycache__' \
  --exclude='malaria-detect/reports' \
  malaria-detect
```

| Piece | Meaning |
|---|---|
| `tar` | Archive tool (tape archive) |
| `-c` | **C**reate a new archive |
| `-z` | Compress with g**z**ip (makes it ~10× smaller) |
| `-f malaria-detect.tar.gz` | Write to this **f**ile |
| `--exclude=...` | Skip these. `.venv` contains Windows binaries that would break on Linux; `.cache` is just download cache |

Why not copy the folder as-is? The virtual environment (`.venv`) was
built for Windows — PyTorch wheels are platform-specific. EC2 rebuilds
everything from `requirements.txt` inside Docker, so the venv is useless
on the server.

## 2. Copy it to the server

```bash
scp -i your-key.pem malaria-detect.tar.gz ec2-user@<EC2_PUBLIC_IP>:~/
```

| Piece | Meaning |
|---|---|
| `scp` | **S**ecure **cop**y — encrypted file transfer over SSH |
| `-i your-key.pem` | Use this **i**dentity (private key) to authenticate |
| `ec2-user@<IP>` | User `ec2-user` (Amazon Linux; use `ubuntu` on Ubuntu AMIs) at the instance's public IP |
| `:~/` | Destination: the home directory on the remote machine |

## 3. Log in

```bash
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>
```

`ssh` = Secure SHell, a remote command line. If the key is rejected,
check its permissions on Windows (Git Bash):
`chmod 600 /c/Users/Admin/your-key.pem`.

## 4. Install Docker on the instance

```bash
sudo yum update -y
sudo yum install -y docker docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER && exit
```

| Command | Meaning |
|---|---|
| `sudo yum update -y` | Update all OS packages. `yum` = the package manager on Amazon Linux; `-y` = auto-answer "yes" to prompts. (Ubuntu: `sudo apt update && sudo apt upgrade -y`) |
| `sudo yum install -y docker docker-compose-plugin` | Install the Docker engine and the `docker compose` plugin |
| `sudo systemctl enable --now docker` | **enable** = start Docker automatically on every boot; `--now` = also start it immediately |
| `sudo usermod -aG docker $USER` | Add your user to the `docker` group so you can run docker without `sudo` (`-aG` = **a**ppend to **G**roup) |
| `&& exit` | Log out — the group membership only applies on the **next** login, so reconnect with the same `ssh` command |

## 5. Add a swap file (prevents OOM during build)

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

| Command | Meaning |
|---|---|
| `fallocate -l 2G /swapfile` | Reserve a 2 GB file. Swap = disk space used as overflow RAM when physical RAM runs out |
| `chmod 600` | Only root may read/write it (security — swap contains memory contents) |
| `mkswap` | Format the file as swap space |
| `swapon` | Activate it now |

The Docker build installs PyTorch (~800 MB of wheels) on a 2-vCPU,
4 GiB machine; without swap it can be killed mid-build. (A t3.large with
8 GB doesn't strictly need this, but it's harmless.)

## 6. Extract and launch

```bash
tar -xzf malaria-detect.tar.gz
cd malaria-detect
docker compose up --build -d
```

| Piece | Meaning |
|---|---|
| `tar -xzf` | E**x**tract the g**z**ip archive to its **f**ile |
| `docker compose up` | Read `docker-compose.yml` and start the service |
| `--build` | **Rebuild the image** from the `Dockerfile` — required, otherwise Docker uses whatever cached/old image exists |
| `-d` | **D**etached — run in the background, keep your shell free |

What happens during the build (takes 10–20 min on 2 vCPUs):

1. Pulls `python:3.12-slim` base image
2. Installs PyTorch **CPU-only wheels** from
   `download.pytorch.org/whl/cpu` (saves ~2 GB of useless CUDA libraries
   — EC2 without a GPU can't use them anyway)
3. Installs FastAPI, torchvision, etc. from `requirements.txt`
4. Copies `app/`, `train/`, `tests/`, `docs/`, `static/` into the image

On first startup the app downloads the 9 MB model weights from the
Hugging Face Hub (`Svetozar1993/LocalMedScan-malaria-mobilenetv2`) and
caches them inside the container. First `/predict` may take ~60 s;
after that it's warm.

## 7. Open port 8000 in the firewall

The API listens on port 8000, but EC2 blocks all inbound traffic by
default.

**AWS Console → EC2 → Instances → your instance → Security group →
Edit inbound rules → Add rule:**

| Type | Port | Source |
|---|---|---|
| Custom TCP | 8000 | Your IP (or a trusted range) |

> ⚠️ The API has **no authentication**. Do **not** open 8000 to
> `0.0.0.0/0` (the whole internet) — anyone could use your instance as
> a free inference server. Restrict the source to your own IP.

## 8. Verify it works

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true}

curl http://localhost:8000/model-info
# {"architecture":"MobileNetV2","parameters":2226434,"device":"cpu", ...}

curl -X POST http://localhost:8000/predict -F "file=@/path/to/cell.png"
```

Then from your PC's browser open:

```
http://<EC2_PUBLIC_IP>:8000/
```

and upload the two MIT images (`MIT-Malaria-Diagnosis-01/02.jpg`) —
both should return **Parasitized** with mode "full-field (patch grid)".

## 9. Day-to-day management

```bash
docker compose logs -f          # follow the service log (Ctrl+C to leave)
docker compose ps               # show running containers
docker compose down             # stop and remove the container
docker compose up -d            # start it again (no rebuild needed)
docker compose up -d --build    # rebuild after code changes
```

To push new code: re-`tar` on the PC → `scp` → `tar -xzf` on the server
→ `docker compose up -d --build`.

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Build killed mid-install | Out of memory — did you create the swap file (step 5)? Or use a t3.large |
| `curl` hangs / connection refused | Security group doesn't allow port 8000 (step 7), or the container isn't up (`docker compose ps`) |
| `docker: permission denied` | You didn't re-login after `usermod` — run `exit` and ssh again |
| No space left on device | Root volume too small — `docker system prune -a` frees old images, or resize the EBS volume |
| First prediction very slow (>60 s) | Normal: weights download + model load. Subsequent requests are ~1.5 s |

## 11. Shut down (stop the billing!)

```bash
docker compose down                      # stop the container
exit                                     # leave the instance
```

Then in the AWS console: **EC2 → Instances → select your instance →
Instance state → Terminate instance**. A stopped instance still bills
for its EBS volume; **terminate** to stop all charges.
