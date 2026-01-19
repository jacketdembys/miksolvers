import torch
print("torch:", torch.__version__)
print("python:", __import__("sys").version.split()[0])
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())
print("cuda available:", torch.cuda.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(1024, 1024, device=device)
y = x @ x
print("device used:", device, "sum:", y.sum().item())