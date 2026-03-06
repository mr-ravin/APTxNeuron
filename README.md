## APTx Neuron 
This repository offers a Python code for the PyTorch implementation of the APTx Neuron and experimentation on MNIST dataset, as introduced in the paper "APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation".

**Paper Title**: APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation

**Author**: [Ravin Kumar](https://mr-ravin.github.io)

**Sources**:
- [Arxiv.org](https://arxiv.org/abs/2507.14270)
- [Research Gate](https://www.researchgate.net/publication/393889376_APTx_Neuron_A_Unified_Trainable_Neuron_Architecture_Integrating_Activation_and_Computation)
- [SSRN](http://dx.doi.org/10.2139/ssrn.5364841)
  
#### Github Repositories: 
- **APTx Neuron** (Pytorch + PyPI Package): [APTx Neuron](https://github.com/mr-ravin/aptx_neuron)
- **APTx Activation Function** (Pytorch + PyPI Package): [APTx Activation Function](https://github.com/mr-ravin/aptx_activation)
- **Experimentation Results with MNIST** (APTx Neuron): [MNIST Experimentation Code](https://github.com/mr-ravin/APTxNeuron)

#### Cite Paper as:
```
Kumar, Ravin. "APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation." arXiv preprint arXiv:2507.14270 (2025).
```
Or,
```
@article{kumar2025aptx,
  title={APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation},
  author={Kumar, Ravin},
  journal={arXiv preprint arXiv:2507.14270},
  year={2025}
}
```
 
---
### APTx Neuron
<b>Abstract</b>: We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the [APTx activation function](https://arxiv.org/abs/2209.06119), thereby eliminating the need for separate activation layers and making the architecture both optimization-efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69\% test accuracy within 11 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and training efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it. Source code is available at [https://github.com/mr-ravin/aptx_neuron](https://github.com/mr-ravin/aptx_neuron).

The APTx Neuron is a novel computational unit that unifies linear transformation and non-linear activation into a single, expressive formulation. Inspired by the parametric APTx activation function, this neuron architecture removes the strict separation between computation and activation, allowing both to be learned as a cohesive entity. It is designed to enhance representational flexibility while reducing architectural redundancy.

#### Mathematical Formulation

Traditionally, a neuron computes the output as:

$y = \phi\left( \sum_{i=1}^{n} w_i x_i + b \right)$

where: 
- $x_i$ are the inputs,
- $w_i$ are the weights,
- $b$ is the bias,
- and $\phi$ is an activation function such as ReLU, Swish, or Mish.


The APTx Neuron merges these components into a unified trainable expression as:

$y = \sum_{i=1}^{n} \left[ (\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i \right] + \delta$

where:
- $x_i$ is the $i$-th input feature,
- $\alpha_i$, $\beta_i$, and $\gamma_i$ are trainable parameters for each input,
- $\delta$ is a trainable scalar bias.

This equation allows the neuron to modulate each input through a learned, per-dimension non-linearity and scaling operation. The term $(\alpha_i + \tanh(\beta_i x_i))$ introduces adaptive gating, and $\gamma_i x_i$ provides multiplicative control.

---
## 📥 Installation
```bash
pip install aptx_neuron
```
or,

```bash
pip install git+https://github.com/mr-ravin/aptx_neuron.git
```
----

## Usage of aptx_neuron Package

<b>1</b>. APTx Neuron-based Layer with all $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable:

The setting `is_alpha_trainable = True` keeps $\alpha_i$ trainable. Each APTx neuron will have $(3n + 1)$ trainable parameters, where $n$ is input dimension. Note: The default value of `is_alpha_trainable` is `True`. 

```
import aptx_neuron
import torch

input_dim = 8  # assuming input vector to be of dimension 8.
output_dim = 2 # assuming output dimension equals 2.

aptx_neuron_layer = aptx_neuron.aptx_layer(input_dim=input_dim, output_dim=output_dim, is_alpha_trainable=True)

batch_size = 3 # assuming batch size is 3
input_tensor = torch.ones((batch_size, input_dim)) # dimension: [3, 8]
output_tensor = aptx_neuron_layer(input_tensor)    # dimension: [3, 2]
```

<b>2</b>. APTx Neuron-based Layer with $\alpha_i=1$ (not trainable); While $\beta_i$, $\gamma_i$, and $\delta$ are trainable:

The setting `is_alpha_trainable = False` makes $\alpha_i$ fixed (non-trainable). Each APTx neuron will then have $(2n + 1)$ trainable parameters, thus reducing memory and training time per epoch. Here, $n$ is input dimension.

```
import aptx_neuron
import torch

input_dim = 8  # assuming input vector to be of dimension 8.
output_dim = 2 # assuming output dimension equals 2.

aptx_neuron_layer = aptx_neuron.aptx_layer(input_dim=input_dim, output_dim=output_dim, is_alpha_trainable=False)  # α_i is fixed (not trainable)

batch_size = 3 # assuming batch size is 3
input_tensor = torch.ones((batch_size, input_dim)) # dimension: [3, 8]
output_tensor = aptx_neuron_layer(input_tensor)    # dimension: [3, 2]
```

<b>3</b>. Single APTx Neuron with trainable $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$:

The setting `is_alpha_trainable = True` keeps $\alpha_i$ trainable. Each APTx neuron will have $(3n + 1)$ trainable parameters, where $n$ is input dimension. Note: The default value of `is_alpha_trainable` is `True`.

```
import aptx_neuron
import torch

input_dim = 8  # assuming input vector to be of dimension 8.

aptx_neuron_unit = aptx_neuron.aptx_neuron(input_dim=input_dim, is_alpha_trainable=True)

batch_size = 3 # assuming batch size is 3
input_tensor = torch.ones((batch_size, input_dim)) # dimension: [3, 8]
output_tensor = aptx_neuron_unit(input_tensor)     # dimension: [3, 1]
```

<b>4</b>. Single APTx Neuron with fixed $\alpha_i=1$ and trainable $\beta_i$, $\gamma_i$, and $\delta$:

The setting `is_alpha_trainable = False` makes $\alpha_i$ fixed (non-trainable). Each APTx neuron will then have $(2n + 1)$ trainable parameters, thus reducing memory and training time per epoch. Here, $n$ is input dimension.

```
import aptx_neuron
import torch

input_dim = 8  # assuming input vector to be of dimension 8.

aptx_neuron_unit = aptx_neuron.aptx_neuron(input_dim=input_dim, is_alpha_trainable=False)  # α_i is fixed (not trainable)

batch_size = 3 # assuming batch size is 3
input_tensor = torch.ones((batch_size, input_dim)) # dimension: [3, 8]
output_tensor = aptx_neuron_unit(input_tensor)    # dimension: [3, 1]
```

> For a layer with `output_dim = m`, the total number of trainable parameters is **m(3n + 1)** when `α_i` is trainable, or **m(2n + 1)** when `α_i = 1` is fixed. Here, **n** is the input dimension (i.e., `input_dim`) and **m** is the number of neurons in the layer (i.e., `output_dim`).


### Working Demonstration

- #### Example: APTx Neuron Unit

```
import torch
import torch.nn as nn
from aptx_neuron import aptx_neuron

# Create single neuron
model = aptx_neuron(input_dim=4)

# Input (batch_size=3, input_dim=4)
x = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.]
])

# Target output (batch_size=3, 1)
target = torch.tensor([
    [1.0],
    [2.0],
    [3.0]
])

# Forward pass
y = model(x) # shape: [3, 1]

# Compute loss
loss_fn = nn.MSELoss()
loss = loss_fn(y, target)

# Backward pass
loss.backward()

print("Output:")
print(y)

print("\nLoss:", loss.item())

if model.alpha.grad is not None:
    print("\nGradient alpha:")
    print(model.alpha.grad)
else:
    print("\nAlpha is not trainable (no gradient).")

print("\nGradient beta:")
print(model.beta.grad)

print("\nGradient gamma:")
print(model.gamma.grad)

print("\nGradient delta:")
print(model.delta.grad)
```

- #### Example: APTx Neuron Layer

```
import torch
import torch.nn as nn
from aptx_neuron import aptx_layer

# Create APTx layer
model = aptx_layer(input_dim=4, output_dim=2)

# Input (batch_size=3, input_dim=4)
x = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.]
])

# Target output (batch_size=3, output_dim=2)
target = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

# Forward pass
y = model(x) # shape: [3, 2]

# Loss
loss_fn = nn.MSELoss()
loss = loss_fn(y, target)

# Backward
loss.backward()

print("Output:")
print(y)

print("\nLoss:", loss.item())

if model.alpha.grad is not None:
    print("\nGradient alpha shape:", model.alpha.grad.shape)
else:
    print("\nAlpha is not trainable (no gradient).")

print("\nGradient beta shape:", model.beta.grad.shape)
print("\nGradient gamma shape:", model.gamma.grad.shape)
print("\nGradient delta shape:", model.delta.grad.shape)
```

----
## Experimentation on MNIST dataset
Run the below command to train an APTx Neuron-based fully-connected neural network on the MNIST dataset. The model will automatically save training `loss` and `accuracy` values in the `./result/` directory.

**Note:** For this MNIST experimentation, we have hardcoded `is_alpha_trainable=True` inside `run.py`.
If you wish to explore the **full flexibility and modularity** of APTx Neuron (including toggling α trainability), we recommend using the official PyPI package:
```bash
pip install aptx-neuron
```
Or, install directly from GitHub:
```bash
pip install git+https://github.com/mr-ravin/aptx_neuron.git
```

<b>1. Model Training</b>

To train the model from scratch (using Python 3.12.11 on cpu):
```python
python3 run.py --total_epoch 20
```
This command will:
- Train the model on MNIST
- Save training and validation loss/accuracy in the ./result/ directory
- Use the default device: cpu

<b>2. Model Inference</b>

To perform inference on the test set using the trained model (reproducibility mode):
```python
python3 run.py --mode infer --device cpu
```
----

### Reproducibility on MNIST dataset

#### 1. Inference on Pre-trained model
   
- ✅ **Test Accuracy**: 96.69%

- 📁 **Model Weights File**: `./weights/aptx_neural_network_11.pt`

- 💻 **Environment**: Python `3.12.11`, Pytorch `>=1.8.0`, and Device: `cpu`
-  **Command**:
``` python3 run.py --mode infer --device cpu --load_model_weights_path ./weights/aptx_neural_network_11.pt ```
   
⚠️ **Important**: The codebase was refactored after training. Use this command to reproduce the test accuracy correctly under the same configuration.
🚀 Note: This result was obtained using default hyperparameters on CPU.
🔧 Further optimization (e.g., learning rate scheduling, weight initialization, architecture tuning, or training on GPU etc.) has the potential to improve accuracy even further.


#### 2. Visualise Model Performance

##### 2.1 Visual analysis of train and test loss values

![image](https://github.com/mr-ravin/APTxNeuron//blob/main/mnist_loss.png?raw=true)

##### 2.2 Visual analysis of train and test accuracy values

![image](https://github.com/mr-ravin/APTxNeuron//blob/main/mnist_accuracy.png?raw=true)

##### 2.3 Training & Evaluation Metrics (APTx Neuron on MNIST)

| **Epoch** | **Train Loss** | **Test Loss** | **Train Accuracy (%)**  | **Test Accuracy (%)**  |
|-----------|----------------|---------------|-------------------------|------------------------|
| 1         | 85.58          | 36.73         | 84.16                   | 89.12                  |
| 2         | 33.27          | 17.82         | 90.16                   | 90.76                  |
| 3         | 19.97          | 28.16         | 91.80                   | 90.82                  |
| 4         | 9.98           | 27.00         | 92.55                   | 90.66                  |
| 5         | 15.28          | 24.45         | 93.59                   | 93.03                  |
| 6         | 13.88          | 9.13          | 97.11                   | 96.33                  |
| 7         | 9.35           | 8.84          | 97.47                   | 95.53                  |
| 8         | 0.00           | 7.73          | 97.38                   | 95.51                  |
| 9         | 1.10           | 9.19          | 97.51                   | 94.47                  |
| 10        | 6.41           | 8.69          | 97.56                   | 95.59                  |
| 11        | 0.00           | 6.81          | 98.75                   | **96.69**              |
| 12        | 0.00           | 6.57          | 99.11                   | 96.53                  |
| 13        | 0.00           | 6.67          | 99.19                   | 96.57                  |
| 14        | 0.00           | 7.29          | 99.21                   | 96.40                  |
| 15        | 0.00           | 6.90          | 99.23                   | 96.46                  |
| 16        | 0.00           | 6.25          | 99.60                   | 96.63                  |
| 17        | 0.00           | 6.21          | 99.77                   | 96.58                  |
| 18        | 0.00           | 6.02          | 99.79                   | 96.65                  |
| 19        | 0.00           | 5.95          | 99.78                   | 96.68                  |
| 20        | 0.00           | 6.13          | **99.81**               | 96.56                  |

> ✅ **Best Test Accuracy:** `96.69%` at **Epoch 11**  
> 📌 Indicates potential for **further improvement** with better optimization or deeper architecture.

----
#### Conclusion
This work introduced the APTx Neuron, a unified, fully trainable neural unit that integrates linear transformation and non-linear activation into a single expression, extending the APTx activation function. By learning per-input parameters $\alpha_i$, $\beta_i$, and $\gamma_i$ for each input $x_i$, and a shared bias term $\delta$ within a neuron, the APTx Neuron removes the need for separate activation layers and enables fine-grained input transformation. The APTx Neuron generalizes traditional neurons and activations, offering greater representational power. Our experiments show that a fully connected APTx Neuron-based feedforward neural network achieves 96.69\% test accuracy on the MNIST dataset within 11 epochs using approximately 332K trainable parameters, demonstrating rapid convergence and high optimization efficiency. This design lays the groundwork for extending APTx Neurons to CNNs and transformers, paving the way for more compact and adaptive deep learning architectures.

----

### 📜 Copyright License
```python
Copyright (c) 2025 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
