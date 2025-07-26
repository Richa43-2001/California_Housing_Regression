import joblib
import torch
import numpy as np

sklearn_model = joblib.load("model.joblib")
coef = sklearn_model.coef_
intercept = sklearn_model.intercept_

# Save unquantized
unquantized_params = {'coef': coef, 'intercept': intercept}
joblib.dump(unquantized_params, 'unquant_params.joblib')

# Manual quantization
scale = 255 / (np.max(coef) - np.min(coef))
q_coef = ((coef - np.min(coef)) * scale).astype(np.uint8)
q_intercept = int((intercept - np.min(coef)) * scale)

quantized_params = {'coef': q_coef, 'intercept': q_intercept, 'scale': scale, 'min': np.min(coef)}
joblib.dump(quantized_params, 'quant_params.joblib')

# Dequantize
dq_coef = q_coef.astype(np.float32) / scale + np.min(coef)
dq_intercept = q_intercept / scale + np.min(coef)

# PyTorch model using quantized weights
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(len(coef), 1)
        self.linear.weight.data = torch.tensor([dq_coef], dtype=torch.float32)
        self.linear.bias.data = torch.tensor([dq_intercept], dtype=torch.float32)

    def forward(self, x):
        return self.linear(x)

model = LinearModel()
data = joblib.load("model.joblib")
print("Quantized model ready.")
