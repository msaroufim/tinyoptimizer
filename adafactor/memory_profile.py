
import math
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import logging
import socket

logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return

def convert_optimizer_state_to_dtype(optimizer, dtype=torch.bfloat16):
    """ Convert all optimizer state tensors to bfloat16. """
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dtype)

class Adafactor(optim.Optimizer):
    """
    Implements Adafactor algorithm.
    Original implementation can be found here
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/optim/adafactor.py

    This implementation is based on:
    `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)

    Note that this optimizer internally adjusts the learning rate
    depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate
    schedule you should set `scale_parameter=False` and
    `relative_step=False`.
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
    ):
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay,
            scale_parameter=scale_parameter,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super(Adafactor, self).__init__(params, defaults)


    def _get_lr(self, param_group, param_state):
        """
        Adafactor can compute its own learning rate based on the step size
        """
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = (
                1e-6 * param_state["step"] if param_group["warmup_init"] else 1e-2
            )
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def _get_options(self, param_group, param_shape):
        """
        Decide whether to use the first moment or not based on the shape of the
        """
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    def _rms(self, tensor):
        """
        Root mean square
        """
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """
        Approximation of exponential moving average of square of gradient
        """
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(
                            grad_shape[:-2] + grad_shape[-1:]
                        ).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(
                        update.mean(dim=-1), alpha=1.0 - beta2t
                    )
                    exp_avg_sq_col.mul_(beta2t).add_(
                        update.mean(dim=-2), alpha=1.0 - beta2t
                    )

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

import numpy as np
# Generate a synthetic dataset
def generate_data(n_samples=100):
    X = np.random.rand(n_samples, 1)
    y = 5 * X + np.random.randn(n_samples, 1) * 0.5 # simple linear relation with noise
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = generate_data()

import torch.nn as nn
# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.non_linear = nn.ReLU()
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear2(self.non_linear(self.linear(x)))

model = SimpleModel().cuda().to(torch.bfloat16)
criterion = nn.MSELoss()

# Function to estimate the size of an object in bytes
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

# This doesn't work
# Calculate memory overhead
import sys
# adafactor_optimizer = Adafactor(model.parameters(), lr=None, scale_parameter=True, relative_step=True, warmup_init=True)
# adam_optimizer = optim.Adam(model.parameters(), lr=1e-3)
# adafactor_size = sys.getsizeof(adafactor_optimizer.state)
# adam_size = sys.getsizeof(adam_optimizer.state)
    # print(f'Adafactor state size: {sizeof_fmt(adafactor_size)}')
    # print(f'Adam state size: {sizeof_fmt(adam_size)}')


def train_model(optimizer_name='adam', convert_to_bf16=False):
    # Make seed deterministic
    torch.manual_seed(0)
    np.random.seed(0)


    # Copy the model to avoid interference between optimizers
    model_copy = SimpleModel()
    
    # Choose optimizer
    if optimizer_name == 'adafactor':
        optimizer = Adafactor(model_copy.parameters(),  scale_parameter=True, relative_step=True, warmup_init=False)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model_copy.parameters(), lr=1e-3)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model_copy.parameters(), lr=1e-3)
    # elif optimizer_name == "adamw":
#     optimizer = optim.AdamW(model_copy.parameters(), lr=1e-3)
    else:
        optimizer = optimizer_name
        # raise ValueError('Unknown optimizer name')
    
    # Convert model to bf16
    if convert_to_bf16:
        convert_optimizer_state_to_dtype(optimizer, dtype=torch)
    epochs = 1000
    loss_history = []

    start_record_memory_history()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model_copy(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        # if epoch % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    export_memory_snapshot()
    stop_record_memory_history()

    
    return loss_history

def basic_plot():

    loss_history_adafactor = train_model(optimizer_name='adafactor', convert_to_bf16=False)
    # loss_history_adam = train_model(optimizer_name='adam', convert_to_bf16=False)
    # loss_history_adamw = train_model(optimizer_name='adamw', convert_to_bf16=False)
    # loss_history_sgd = train_model(optimizer_name='sgd', convert_to_bf16=False)


    # Plotting
    # plt.plot(loss_history_adafactor, label='Adafactor')
    # # plt.plot(loss_history_adam, label='Adam')
    # # plt.plot(loss_history_sgd, label='SGD')
    # # plt.plot(loss_history_adamw, label='AdamW')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Loss History')
    # plt.legend()
    # plt.show()

def grid_search():
    import itertools



    # Define the hyperparameter grid
    # lr_options = [None, 1e-1, 1e-2, 1e-3]
    eps_options = [(1e-30, 1e-3), (1e-10, 1e-3), (1e-8, 1e-2)]
    clip_threshold_options = [1.0, 0.5, 2.0]
    decay_rate_options = [-0.8, -0.5, -0.3]
    beta1_options = [None, 0.9, 0.99]
    weight_decay_options = [0.0, 0.01, 0.001]

    # Perform grid search
    results = {}
    for eps, clip_threshold, decay_rate, beta1, weight_decay in itertools.product(
        eps_options, clip_threshold_options, decay_rate_options, beta1_options, weight_decay_options
    ):
        optimizer = Adafactor(
            params=model.parameters(),  # Replace with your model parameters
            # lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            beta1=beta1,
            weight_decay=weight_decay
        )
        loss_history = train_model(optimizer)  # Set your epochs
        results[(eps, clip_threshold, decay_rate, beta1, weight_decay)] = loss_history

    # Plot the convergence of each hyperparameter set
    for params, loss_history in results.items():
        plt.plot(loss_history, label=params)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Hyperparameter Tuning Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

if __name__ == "__main__":
    basic_plot()
    # grid_search()