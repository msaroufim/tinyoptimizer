# Python implementation
class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [0.0 for _ in parameters]
        self.v = [0.0 for _ in parameters]
        self.t = 0
        self.parameters = parameters

    def step(self, grads):
        self.t += 1
        m_hat = [0.0 for _ in self.parameters]
        v_hat = [0.0 for _ in self.parameters]

        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat[i] = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat[i] = self.v[i] / (1 - self.beta2 ** self.t)

            self.parameters[i] -= self.lr * m_hat[i] / (v_hat[i] ** 0.5 + self.epsilon)

        return self.parameters


# PyTorch implementation
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0
        self.params = params

    def step(self):
        self.t += 1
        with torch.no_grad():
            for param, m, v in zip(self.params, self.m, self.v):
                if param.grad is None:
                    continue
                grad = param.grad.data

                m.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                param.addcdiv_(m_hat, v_hat.sqrt().add_(self.epsilon), value=-self.lr)
