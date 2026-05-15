import torch
from safetensors.torch import save_file
from pathlib import Path


class TinyMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 3, bias=True)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(3, 2, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def main():
    torch.manual_seed(1337)
    model = TinyMlp().to(dtype=torch.float32)

    x = torch.randn(5, 4, dtype=torch.float32)
    target = torch.randn(5, 2, dtype=torch.float32)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-2,
    )

    output = model(x)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()

    tensors = {}
    for name, param in model.named_parameters():
        safe_name = name.replace(".", "_")
        tensors[f"param_{safe_name}"] = param.detach().cpu().clone()
        tensors[f"grad_{safe_name}"] = param.grad.detach().cpu().clone()

    optimizer.step()

    for name, param in model.named_parameters():
        safe_name = name.replace(".", "_")
        tensors[f"updated_{safe_name}"] = param.detach().cpu().clone()

    out_path = Path(__file__).parent / "reference.safetensors"
    save_file(tensors, out_path)


if __name__ == "__main__":
    main()
