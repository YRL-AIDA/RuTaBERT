# CTA model

## Dataset
Before training model you need to move data_*.csv files from dataset to `data` directory.

## Docker

### Requirements:
- Docker
- NVIDIA driver
- [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)

### Docker installation (Ubuntu)
Set up Docker's apt repository.

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

To install the latest version, run:
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify:
```
sudo docker run hello-world
```

### Nvidia container toolkit installation (Ubuntu)
Configure the production repository:

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Optionally, configure the repository to use experimental packages:
```
sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Update the packages list from the repository:
```
sudo apt-get update
```

Install the NVIDIA Container Toolkit packages:
```
sudo apt-get install -y nvidia-container-toolkit
```

Configure the container runtime by using the nvidia-ctk command:
```
sudo nvidia-ctk runtime configure --runtime=docker
```
The nvidia-ctk command modifies the `/etc/docker/daemon.json` file on the host. The file is updated so that Docker can use the NVIDIA Container Runtime.

Restart the Docker daemon:
```
sudo systemctl restart docker
```

### Build image
```
sudo docker build -t <image_name> .
```

### Run image
```
sudo docker run -d --runtime=nvidia --gpus=all \
    --mount source=<volume_logs_name>,target=/app/cta/logs \
    --mount source=<volume_checkpoints_name>,target=/app/cta/checkpoints \
    <image_name>
```

### Move models and logs from container after training
```
sudo cp -r /var/lib/docker/volumes/cta_checkpoints/_data ./checkpoints
```

```
sudo cp -r /var/lib/docker/volumes/cta_logs/_data ./logs
```
