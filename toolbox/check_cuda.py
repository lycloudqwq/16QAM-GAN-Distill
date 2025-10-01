import torch


def check_cuda():
    print("Torch version:", torch.__version__)

    if torch.cuda.is_available():
        print("✅ CUDA is available, current GPU:", torch.cuda.get_device_name(0))
    else:
        print("❌ CUDA not available")


if __name__ == "__main__":
    check_cuda()
