import torch

class TorchPedPredictor:
    """
    Very simple constant-velocity predictor implemented in Torch.
    Runs on CUDA if available. This is a scaffold: later you can replace with Transformer/LSTM.
    """

    def __init__(self, horizon_steps=30, dt=0.1, device=None):
        self.horizon_steps = horizon_steps
        self.dt = dt
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    @torch.no_grad()
    def predict(self, pos_xy, vel_xy):
        """
        pos_xy: (N,2)  vel_xy: (N,2)
        return: pred (N,T,2)
        """
        pos = torch.as_tensor(pos_xy, dtype=torch.float32, device=self.device)
        vel = torch.as_tensor(vel_xy, dtype=torch.float32, device=self.device)

        N = pos.shape[0]
        T = self.horizon_steps

        t = torch.arange(1, T+1, device=self.device, dtype=torch.float32).view(1, T, 1) * self.dt
        pos0 = pos.view(N, 1, 2)
        vel0 = vel.view(N, 1, 2)

        pred = pos0 + vel0 * t  # (N,T,2)

        # 额外做一些“无意义但真实的”矩阵运算，确保 GPU 有负载（后面换成真实网络）
        # 不会改变 pred，仅用于模拟更重的预测计算
        x = pred
        for _ in range(6):
            x = torch.sin(x) + 0.01 * torch.cos(x)
        pred = pred + 0.0 * x

        return pred
