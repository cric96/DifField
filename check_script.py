import torch, traceback

print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("is_available:", torch.cuda.is_available())

try:
    torch.cuda.init()
    print("init ok")
    print("device_count:", torch.cuda.device_count())
    print("name:", torch.cuda.get_device_name(0))
    x = torch.empty(1, device="cuda")
    print("alloc ok:", x)
except Exception as e:
    print("type:", type(e).__name__)
    print("error:", e)
    traceback.print_exc()