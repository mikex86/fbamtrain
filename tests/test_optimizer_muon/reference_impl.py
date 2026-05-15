import torch
from safetensors.torch import save_file
from pathlib import Path


def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # Normalize in FP32 for stability, then cast to BF16 for the NS loop.
    X = G.float()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    X = X.bfloat16()
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyMlp().to(device=device, dtype=torch.float32)

    x = torch.randn(5, 4, device=device, dtype=torch.float32)
    target = torch.randn(5, 2, device=device, dtype=torch.float32)

    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adam_params = [p for p in model.parameters() if p.ndim < 2]

    param_groups = []
    if muon_params:
        param_groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=1.0e-3,
                momentum=0.95,
                weight_decay=1.0e-2,
            )
        )
    if adam_params:
        param_groups.append(
            dict(
                params=adam_params,
                use_muon=False,
                lr=1.0e-3,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                weight_decay=1.0e-2,
            )
        )

    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

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
