---
jupyter:
  accelerator: GPU
  colab:
    authorship_tag: ABX9TyPiZqHPF/YamI5YlikNi4KW
    include_colab_link: true
    name: 01_pytorch_workflow_video.ipynb
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.16
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" colab_type="text" id="view-in-github">

<a href="https://colab.research.google.com/github/mrdbourke/pytorch-deep-learning/blob/main/video_notebooks/01_pytorch_workflow_video.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

</div>

<div class="cell markdown">

# 1. <a id='toc1_'></a>[PyTorch Workflow](#toc0_)

Let's explore a an example PyTorch end-to-end workflow.

Resources:

-   Ground truth notebook -
    <https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb>
-   Book version of notebook -
    <https://www.learnpytorch.io/01_pytorch_workflow/>
-   Ask a question -
    <https://github.com/mrdbourke/pytorch-deep-learning/discussions>

</div>

<div class="cell markdown">

**Table of contents**<a id='toc0_'></a>

-   1.  [PyTorch Workflow](#toc1_)  

    -   1.1. [Check the software and OS version](#toc1_1_)  
    -   1.2. [Deep-Learning-API](#toc1_2_)  
    -   1.3. [Data (preparing and loading)](#toc1_3_)
        -   1.3.1. [Splitting data into training and test sets (one of
            the most important concepts in machine learning in
            general)](#toc1_3_1_)  
        -   1.3.2. [Build the plot_predictions function](#toc1_3_2_)  
    -   1.4. [Build model](#toc1_4_)
        -   1.4.1. [PyTorch model building essentials](#toc1_4_1_)  
        -   1.4.2. [Checking the contents of our PyTorch
            model](#toc1_4_2_)  
        -   1.4.3. [Making prediction using
            `torch.inference_mode()`](#toc1_4_3_)  
        -   1.4.4. [Plot prediction to compare between prediction and
            Training data](#toc1_4_4_)  
    -   1.5. [Train model](#toc1_5_)
        -   1.5.1. [Building a training loop (and a testing loop) in
            PyTorch](#toc1_5_1_)
            -   1.5.1.1. [Plot the loss curves](#toc1_5_1_1_)  
        -   1.5.2. [Plot prediction to compare between prediction and
            Training data](#toc1_5_2_)  
    -   1.6. [Saving a model in PyTorch](#toc1_6_)  
    -   1.7. [Loading a PyTorch model](#toc1_7_)  
    -   1.8. [Putting it all together](#toc1_8_)
        -   1.8.1. [Create Data](#toc1_8_1_)  
        -   1.8.2. [Building a PyTorch Linear model](#toc1_8_2_)
            -   1.8.2.1. [Let the model use GPU](#toc1_8_2_1_)  
        -   1.8.3. [Training](#toc1_8_3_)
            -   1.8.3.1. [Write Loss function and
                Optimizer](#toc1_8_3_1_)  
            -   1.8.3.2. [Write Training loop and Testing
                loop](#toc1_8_3_2_)  
        -   1.8.4. [Making and evaluating predictions](#toc1_8_4_)  
        -   1.8.5. [Plot prediction to compare between prediction and
            Training data](#toc1_8_5_)  
        -   1.8.6. [Saving & loading a trained model](#toc1_8_6_)  
    -   1.9. [Exercises & Extra-curriculum](#toc1_9_)

<!-- vscode-jupyter-toc-config
	numbering=true
	anchor=true
	flat=false
	minLevel=1
	maxLevel=6
	/vscode-jupyter-toc-config -->
<!-- THIS CELL WILL BE REPLACED ON TOC UPDATE. DO NOT WRITE YOUR TEXT IN THIS CELL -->

</div>

<div class="cell markdown">

## 1.1. <a id='toc1_1_'></a>[Check the software and OS version](#toc0_)

</div>

<div class="cell code" execution_count="148">

``` python
# Add timestamp
import datetime
print(f"Notebook last run (end-to-end): {datetime.datetime.now()}")
```

<div class="output stream stdout">

    Notebook last run (end-to-end): 2023-06-02 17:50:05.774708

</div>

</div>

<div class="cell code" execution_count="149">

``` python
# Check to see if we're using a GPU
!nvidia-smi
```

<div class="output stream stdout">

    Fri Jun  2 17:50:05 2023       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 531.68                 Driver Version: 531.68       CUDA Version: 12.1     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                      TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce RTX 2060       WDDM | 00000000:01:00.0  On |                  N/A |
    | 32%   36C    P8               23W / 184W|   1673MiB / 12288MiB |     22%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
                                                                                             
    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |    0   N/A  N/A       920    C+G   ...oogle\Chrome\Application\chrome.exe    N/A      |
    |    0   N/A  N/A      2648    C+G   C:\Windows\explorer.exe                   N/A      |
    |    0   N/A  N/A      4508    C+G   ...1.0_x64__8wekyb3d8bbwe\Video.UI.exe    N/A      |
    |    0   N/A  N/A      4544    C+G   ...rnzvtm6\SafeInCloud\SafeInCloud.exe    N/A      |
    |    0   N/A  N/A      5788    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe    N/A      |
    |    0   N/A  N/A      7096    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe    N/A      |
    |    0   N/A  N/A      8620    C+G   ...64__8wekyb3d8bbwe\CalculatorApp.exe    N/A      |
    |    0   N/A  N/A     10940    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe    N/A      |
    |    0   N/A  N/A     11584    C+G   ...02.0_x86__zpdnekdrzrea0\Spotify.exe    N/A      |
    |    0   N/A  N/A     12048    C+G   ...b3d8bbwe\Microsoft.Media.Player.exe    N/A      |
    |    0   N/A  N/A     12416    C+G   ..._8wekyb3d8bbwe\Microsoft.Photos.exe    N/A      |
    |    0   N/A  N/A     12960    C+G   ...les\Microsoft OneDrive\OneDrive.exe    N/A      |
    |    0   N/A  N/A     14280      C   ...1__cuda_11.8__python_3.9\python.exe    N/A      |
    |    0   N/A  N/A     14812    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe    N/A      |
    |    0   N/A  N/A     15184    C+G   ...Programs\Microsoft VS Code\Code.exe    N/A      |
    |    0   N/A  N/A     15232    C+G   ...5n1h2txyewy\ShellExperienceHost.exe    N/A      |
    |    0   N/A  N/A     15268    C+G   ...302.5.0_x64__8wekyb3d8bbwe\Time.exe    N/A      |
    |    0   N/A  N/A     15500    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe    N/A      |
    |    0   N/A  N/A     17120    C+G   ...2txyewy\StartMenuExperienceHost.exe    N/A      |
    |    0   N/A  N/A     18484    C+G   ...61.0_x64__8wekyb3d8bbwe\GameBar.exe    N/A      |
    |    0   N/A  N/A     18696    C+G   ...siveControlPanel\SystemSettings.exe    N/A      |
    +---------------------------------------------------------------------------------------+

</div>

</div>

<div class="cell markdown">

    Tue May 30 11:05:02 2023       
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 531.68                 Driver Version: 531.68       CUDA Version: 12.1     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                      TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  NVIDIA GeForce RTX 2060       WDDM | 00000000:01:00.0  On |                  N/A |
    | 32%   38C    P8               22W / 184W|    782MiB / 12288MiB |     23%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+

</div>

<div class="cell code" execution_count="150">

``` python
# Check your software and os version
import os
print("posix = Linux, nt = windows, java = java")
print("your os name is",os.name)

import platform
print("your os is ",platform.system()+platform.release())

from platform import python_version
print("python version is",python_version())

import torch
print("Pytorch version is",torch.__version__)

print("Are we using a GPU?",torch.cuda.is_available())
```

<div class="output stream stdout">

    posix = Linux, nt = windows, java = java
    your os name is nt
    your os is  Windows10
    python version is 3.9.16
    Pytorch version is 2.0.1+cu118
    Are we using a GPU? True

</div>

</div>

<div class="cell code" execution_count="151"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="z_n_NlLzFwEN" outputId="0f9c66d7-e8af-4020-d53c-17c2e1ede55f">

``` python
what_were_covering = {1: "data (prepare and load)",
                      2: "build model",
                      3: "fitting the model to data (training)",
                      4: "making predictions and evaluting a model (inference)",
                      5: "saving and loading a model",
                      6: "putting it all together"}

what_were_covering
```

<div class="output execute_result" execution_count="151">

    {1: 'data (prepare and load)',
     2: 'build model',
     3: 'fitting the model to data (training)',
     4: 'making predictions and evaluting a model (inference)',
     5: 'saving and loading a model',
     6: 'putting it all together'}

</div>

</div>

<div class="cell code" execution_count="152"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:35}"
id="OJN3I__OGWOe" outputId="1e270c8b-bbb2-4901-b1c7-bcf3e0f1e9f5">

``` python
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks 
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__
```

<div class="output execute_result" execution_count="152">

    '2.0.1+cu118'

</div>

</div>

<div class="cell markdown">

## 1.2. <a id='toc1_2_'></a>[Deep-Learning-API](#toc0_)

</div>

<div class="cell markdown">

If you want learn more about this API, check this
:<https://github.com/chioujryu/Deep-Learning-API/tree/main>

</div>

<div class="cell code" execution_count="153">

``` python
# ==================for linux command==================
# import wget
# import pathlib
# PATH = "https://github.com/chioujryu/Deep-Learning-API/raw/main/helper_functions_generic.py"
# LAST_PATH = pathlib.PurePath(PATH).name
# if os.path.exists(LAST_PATH) != True:
#     !wget $PATH

# ==================for windows command==================
import wget
import pathlib
PATH = "https://github.com/chioujryu/Deep-Learning-API/raw/main/helper_functions_generic.py"
LAST_PATH = pathlib.PurePath(PATH).name
# print(last_part)  # 👉️ 'last'

if os.path.exists(LAST_PATH) != True:
    wget.download(PATH)
```

</div>

<div class="cell code" execution_count="154">

``` python
from helper_functions_generic import list_directory_tree_structure
```

</div>

<div class="cell markdown">

## 1.3. <a id='toc1_3_'></a>[Data (preparing and loading)](#toc0_)

Data can be almost anything... in machine learning.

-   Excel speadsheet
-   Images of any kind
-   Videos (YouTube has lots of data...)
-   Audio like songs or podcasts
-   DNA
-   Text

Machine learning is a game of two parts:

1.  Get data into a numerical representation.
2.  Build a model to learn patterns in that numerical representation.

To showcase this, let's create some *known* data using the linear
regression formula.

We'll use a linear regression formula to make a straight line with
*known* **parameters**.

</div>

<div class="cell code" execution_count="155"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="5hCumNpHHCTU" outputId="ca51def4-8b84-4b2a-80f8-2da542907aed">

``` python
# Create *known* parameters
weight = 0.7
bias = 0.3

# Create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will pop up when training
y = weight * X + bias 

print("X look like = \n", X[:10], type(X), "\n" , "y look like = \n", y[:10], type(y))
```

<div class="output stream stdout">

    X look like = 
     tensor([[0.0000],
            [0.0200],
            [0.0400],
            [0.0600],
            [0.0800],
            [0.1000],
            [0.1200],
            [0.1400],
            [0.1600],
            [0.1800]]) <class 'torch.Tensor'> 
     y look like = 
     tensor([[0.3000],
            [0.3140],
            [0.3280],
            [0.3420],
            [0.3560],
            [0.3700],
            [0.3840],
            [0.3980],
            [0.4120],
            [0.4260]]) <class 'torch.Tensor'>

</div>

</div>

<div class="cell code" execution_count="156"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="lrPJV_XkJgRT" outputId="a7c152d0-59d6-455f-c61d-4445f1311014">

``` python
len(X), len(y)
```

<div class="output execute_result" execution_count="156">

    (50, 50)

</div>

</div>

<div class="cell markdown">

### 1.3.1. <a id='toc1_3_1_'></a>[Splitting data into training and test sets (one of the most important concepts in machine learning in general)](#toc0_)

Let's create a training and test set with our data.

</div>

<div class="cell code" execution_count="157"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vpMm7mp_KtNH" outputId="ff199d0c-6974-47f7-8ba7-e51b7df4c6b6">

``` python
# Create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:] 

len(X_train), len(y_train), len(X_test), len(y_test)

# You also can use train_test_split
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
```

<div class="output execute_result" execution_count="157">

    (40, 40, 10, 10)

</div>

</div>

<div class="cell markdown" id="AqArrYcENbhp">

How might we better visualize our data?

This is where the data explorer's motto comes in!

"Visualize, visualize, visualize!"

</div>

<div class="cell markdown">

### 1.3.2. <a id='toc1_3_2_'></a>[Build the plot_predictions function](#toc0_)

</div>

<div class="cell code" execution_count="158" id="Bgb1fH7FL0O8">

``` python
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  
  # Show the legend
  plt.legend(prop={"size": 14});

  # you can check the usage from here: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html
```

</div>

<div class="cell code" execution_count="159"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="8yWmPL7gMfPE" outputId="d34d9daa-4ffe-4f22-8ee0-44fa31117490">

``` python
plot_predictions();
```

<div class="output display_data">

![](7f448feaa430fa2daa35c9613fed06c7de4504c2.png)

</div>

</div>

<div class="cell markdown">

## 1.4. <a id='toc1_4_'></a>[Build model](#toc0_)

Our first PyTorch model!

This is very exciting... let's do it!

Because we're going to be building classes throughout the course, I'd
recommend getting familiar with OOP in Python, to do so you can use the
following resource from Real Python:
<https://realpython.com/python3-object-oriented-programming/>

What our model does:

-   Start with random values (weight & bias)
-   Look at training data and adjust the random values to better
    represent (or get closer to) the ideal values (the weight & bias
    values we used to create the data)

How does it do so?

Through two main algorithms:

1.  Gradient descent - <https://youtu.be/IHZwWFHWa-w>
2.  Backpropagation - <https://youtu.be/Ilg3gGewQ5U>

</div>

<div class="cell code" execution_count="160" id="qirhP4VUkOky">

``` python
from torch import nn

# Create linear regression model class
class LinearRegressionModel(nn.Module): # <- almost everything in PyTorch inherhits from nn.Module
  # https://pytorch.org/docs/stable/generated/torch.nn.Module.html
  def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1, # <- start with a random weight and try to adjust it to the ideal weight
                                            requires_grad=True, # <- can this parameter be updated via gradient descent?
                                            dtype=torch.float)) # <- PyTorch loves the datatype torch.float32
    
    self.bias = nn.Parameter(torch.randn(1, # <- start with a random bias and try to adjust it to the ideal bias
                                         requires_grad=True, # <- can this parameter be updated via gradient descent?
                                         dtype=torch.float)) # <- PyTorch loves the datatype torch.float32 
    
  # Forward method to define the computation in the model
  # `x: torch.Tensor` 意思是 x 的型別必須要是 torch.Tensor
  def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data
    return self.weights * x + self.bias # this is the linear regression formula
```

</div>

<div class="cell markdown">

### 1.4.1. <a id='toc1_4_1_'></a>[PyTorch model building essentials](#toc0_)

-   torch.nn - contains all of the buildings for computational graphs (a
    neural network can be considered a computational graph)
-   torch.nn.Parameter - what parameters should our model try and learn,
    often a PyTorch layer from torch.nn will set these for us
-   torch.nn.Module - The base class for all neural network modules, if
    you subclass it, you should overwrite forward()
-   torch.optim - this where the optimizers in PyTorch live, they will
    help with gradient descent
-   def forward() - All nn.Module subclasses require you to overwrite
    forward(), this method defines what happens in the forward
    computation

See more of these essential modules via the PyTorch cheatsheet -
<https://pytorch.org/tutorials/beginner/ptcheat.html>

</div>

<div class="cell markdown">

### 1.4.2. <a id='toc1_4_2_'></a>[Checking the contents of our PyTorch model](#toc0_)

Now we've created a model, let's see what's inside...

So we can check our model parameters or what's inside our model using
`.parameters()`.

</div>

<div class="cell code" execution_count="161">

``` python
# 如果要固定 torch.rand(1) 的值，前面都要加上 torch.manual_seed(42)
# torch.manual_seed(42)
torch.rand(1)
```

<div class="output execute_result" execution_count="161">

    tensor([0.3829])

</div>

</div>

<div class="cell code" execution_count="162"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="0737rQGNtDxP" outputId="2a477df1-d234-4db1-c25d-f6eb734cbf86">

``` python
# Create a random seed
# 因為在model裡面有torch.rand
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

# Check out the parameters
list(model_0.parameters())
```

<div class="output execute_result" execution_count="162">

    [Parameter containing:
     tensor([0.3367], requires_grad=True),
     Parameter containing:
     tensor([0.1288], requires_grad=True)]

</div>

</div>

<div class="cell code" execution_count="163"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="hdzvifGftWYZ" outputId="983cdf2a-c582-4fbf-e8d8-9060bb32bb30">

``` python
# List named parameters
model_0.state_dict()
```

<div class="output execute_result" execution_count="163">

    OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])

</div>

</div>

<div class="cell code" execution_count="164">

``` python
# 最一開始設定的weight跟bias
weight, bias
```

<div class="output execute_result" execution_count="164">

    (0.7, 0.3)

</div>

</div>

<div class="cell markdown">

### 1.4.3. <a id='toc1_4_3_'></a>[Making prediction using `torch.inference_mode()`](#toc0_)

To check our model's predictive power, let's see how well it predicts
`y_test` based on `X_test`.

When we pass data through our model, it's going to run it through the
`forward()` method.

</div>

<div class="cell code" execution_count="165"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="j_nRrqGMwm0N" outputId="f9cbe561-5994-4f99-e4e2-96f70b06fbb8">

``` python
y_preds = model_0(X_test)
y_preds
```

<div class="output execute_result" execution_count="165">

    tensor([[0.3982],
            [0.4049],
            [0.4116],
            [0.4184],
            [0.4251],
            [0.4318],
            [0.4386],
            [0.4453],
            [0.4520],
            [0.4588]], grad_fn=<AddBackward0>)

</div>

</div>

<div class="cell code" execution_count="166"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="qCASe_bouVKL" outputId="cdcc4351-3173-44d2-b854-7bdecedcd0b1">

``` python
# Make predictions with model
# 推理模式API ( Inference Mode API ) 可以显著加速推理工作负载的速度，
# 同时保持安全，并确保永远不会计算不正确的梯度. 在不需要 autograd 时，其提供了最好的性能.
with torch.inference_mode():
  y_preds = model_0(X_test)
  

# # You can also do something similar with torch.no_grad(), however, torch.inference_mode() is preferred
# with torch.no_grad():
#   y_preds = model_0(X_test)

y_preds
```

<div class="output execute_result" execution_count="166">

    tensor([[0.3982],
            [0.4049],
            [0.4116],
            [0.4184],
            [0.4251],
            [0.4318],
            [0.4386],
            [0.4453],
            [0.4520],
            [0.4588]])

</div>

</div>

<div class="cell markdown" id="HYHvIyDsxL65">

See more on inference mode here -
<https://twitter.com/PyTorch/status/1437838231505096708?s=20&t=cnKavO9iTgwQ-rfri6u7PQ>

</div>

<div class="cell code" execution_count="167"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="FVREWa_BvzI0" outputId="a89181a5-a12c-4f52-ef03-e01943da9561">

``` python
# 因為weight跟bias不一樣，所以會跟y_preds不一樣
y_test
```

<div class="output execute_result" execution_count="167">

    tensor([[0.8600],
            [0.8740],
            [0.8880],
            [0.9020],
            [0.9160],
            [0.9300],
            [0.9440],
            [0.9580],
            [0.9720],
            [0.9860]])

</div>

</div>

<div class="cell markdown">

### 1.4.4. <a id='toc1_4_4_'></a>[Plot prediction to compare between prediction and Training data](#toc0_)

</div>

<div class="cell code" execution_count="168"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="331WoqFewSnl" outputId="059f3ceb-bc54-42b4-88c5-433930e8a50d">

``` python
plot_predictions(predictions=y_preds)
```

<div class="output display_data">

![](6a7e97f269487c1df25b142871dcd4a95d6528cc.png)

</div>

</div>

<div class="cell markdown">

## 1.5. <a id='toc1_5_'></a>[Train model](#toc0_)

The whole idea of training is for a model to move from some *unknown*
parameters (these may be random) to some *known* parameters.

Or in other words from a poor representation of the data to a better
representation of the data.

One way to measure how poor or how wrong your models predictions are is
to use a loss function.

-   Note: Loss function may also be called cost function or criterion in
    different areas. For our case, we're going to refer to it as a loss
    function.

Things we need to train:

-   **Loss function:** A function to measure how wrong your model's
    predictions are to the ideal outputs, lower is better. -
    <https://pytorch.org/docs/stable/nn.html#loss-functions>
-   **Optimizer:** Takes into account the loss of a model and adjusts
    the model's parameters (e.g. weight & bias in our case) to improve
    the loss function -
    <https://pytorch.org/docs/stable/optim.html#module-torch.optim>
    -   Inside the optimizer you'll often have to set two parameters:
        -   `params` - the model parameters you'd like to optimize, for
            example `params=model_0.parameters()`
        -   `lr` (learning rate) - the learning rate is a hyperparameter
            that defines how big/small the optimizer changes the
            parameters with each step (a small `lr` results in small
            changes, a large `lr` results in large changes)

And specifically for PyTorch, we need:

-   A training loop
-   A testing loop

</div>

<div class="cell code" execution_count="169"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="gR7Xy8FqQ1x1" outputId="85ea9234-292b-4269-c450-3a9088e3d65a">

``` python
list(model_0.parameters())
```

<div class="output execute_result" execution_count="169">

    [Parameter containing:
     tensor([0.3367], requires_grad=True),
     Parameter containing:
     tensor([0.1288], requires_grad=True)]

</div>

</div>

<div class="cell code" execution_count="170"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="MeA8r0whRzVM" outputId="3bf7b029-3a6d-4f4d-cb37-e493fff66f16">

``` python
# Check out our model's parameters (a parameter is a value that the model sets itself)
model_0.state_dict()
```

<div class="output execute_result" execution_count="170">

    OrderedDict([('weights', tensor([0.3367])), ('bias', tensor([0.1288]))])

</div>

</div>

<div class="cell code" execution_count="171" id="FhpnOr3vR5jI">

``` python
# =====================Setup a loss function for linear regression=====================
# use nn.L1Loss()
# nn.L1Loss() = torch.mean(torch.abs(y_pred-y_test))
loss_function = nn.L1Loss()

# use nn.MSELoss(), nn.MSELoss() = L2 Loss
# loss_fn = nn.MSELoss()

# =====================Setup an optimizer (stochastic gradient descent)=====================
# SGD Optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # we want to optimize the parameters present in our model
                            # lr = learning rate = possibly the most important hyperparameter you can set
                            # The larger the learning rate, the greater the magnitude of change in the weights and biases.
                            lr=0.001,    
                            momentum=0.9)

# Adam Optimizer
# optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.0001)
```

</div>

<div class="cell markdown" id="zR0wku4LUzr7">

> **Q:** Which loss function and optimizer should I use?
>
> **A:** This will be problem specific. But with experience, you'll get
> an idea of what works and what doesn't with your particular problem
> set.
>
> For example, for a regression problem (like ours), a loss function of
> `nn.L1Loss()` and an optimizer like `torch.optim.SGD()` will suffice.
>
> But for a classification problem like classifying whether a photo is
> of a dog or a cat, you'll likely want to use a loss function of
> `nn.BCELoss()` (binary cross entropy loss).

</div>

<div class="cell markdown">

### 1.5.1. <a id='toc1_5_1_'></a>[Building a training loop (and a testing loop) in PyTorch](#toc0_)

A couple of things we need in a training loop:

1.  Loop through the data and do...
2.  Forward pass (this involves data moving through our model's
    `forward()` functions) to make predictions on data - also called
    forward propagation
3.  Calculate the loss (compare forward pass predictions to ground truth
    labels)
4.  Optimizer zero grad
5.  Loss backward - move backwards through the network to calculate the
    gradients of each of the parameters of our model with respect to the
    loss (**backpropagation** -
    <https://www.youtube.com/watch?v=tIeHLnjs5U8>)
6.  Optimizer step - use the optimizer to adjust our model's parameters
    to try and improve the loss (**gradient descent** -
    <https://youtu.be/IHZwWFHWa-w>)

</div>

<div class="cell code" execution_count="172">

``` python
torch.manual_seed(42)

# An epoch is one loop through the data... (this is a hyperparameter because we've set it ourselves)
epochs = 1

# ================Training================
# 0. Loop through the data
for epoch in range(epochs): 
    # Set the model to training mode
    model_0.train() # `train mode` in PyTorch sets all parameters that require gradients to require gradients 

    # 1. Forward pass
    y_pred = model_0(X_train)

    # 2. Calculate the loss
    loss = loss_function(y_pred, y_train)

    # 3. Optimizer zero grad
    optimizer.zero_grad()  # reset

    # 4. Perform backpropagation on the loss with respect to the parameters of the model (calculate gradients of each parameter)
    loss.backward()

    # 5. Step the optimizer (perform gradient descent)
    optimizer.step() # by default how the optimizer changes will acculumate through the loop so... we have to zero them above in step 3 for the next iteration of the loop

    ### Testing
    model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
    print(model_0.state_dict())
```

<div class="output stream stdout">

    OrderedDict([('weights', tensor([0.3371])), ('bias', tensor([0.1298]))])

</div>

</div>

<div class="cell code" execution_count="173"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="TV8WOxYPaTlP" outputId="e05110b7-2289-48c9-e99c-cfca09e0f9b3">

``` python
torch.manual_seed(42)

# An epoch is one loop through the data... (this is a hyperparameter because we've set it ourselves)
epochs = 200

# Track different values
epoch_count = [] 
loss_values = []
test_loss_values = [] 

# ================Training================
# 0. Loop through the data
for epoch in range(epochs): 
  # Set the model to training mode
  model_0.train() # `train mode` in PyTorch sets all parameters that require gradients to require gradients 
 
  # 1. Forward pass
  y_pred = model_0(X_train)

  # 2. Calculate the loss
  loss = loss_function(y_pred, y_train)

  # 3. Optimizer zero grad
  optimizer.zero_grad()  # reset

  # 4. Perform backpropagation on the loss with respect to the parameters of the model (calculate gradients of each parameter)
  loss.backward()

  # 5. Step the optimizer (perform gradient descent)
  optimizer.step() # by default how the optimizer changes will acculumate through the loop so... we have to zero them above in step 3 for the next iteration of the loop

  ### Testing
  model_0.eval() # turns off different settings in the model not needed for evaluation/testing (dropout/batch norm layers)
  with torch.inference_mode(): # turns off gradient tracking & a couple more things behind the scenes - https://twitter.com/PyTorch/status/1437838231505096708?s=20&t=aftDZicoiUGiklEP179x7A
  # with torch.no_grad(): # you may also see torch.no_grad() in older PyTorch code
    # 1. Do the forward pass 
    test_pred = model_0(X_test)

    # 2. Calculate the loss
    test_loss = loss_function(test_pred, y_test)

  # Print out what's happenin'
  if epoch % 10 == 0:
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    # Print out model state_dict()
    print(model_0.state_dict())
```

<div class="output stream stdout">

    Epoch: 0 | Loss: 0.3117292523384094 | Test loss: 0.4906295835971832
    OrderedDict([('weights', tensor([0.3378])), ('bias', tensor([0.1317]))])
    Epoch: 10 | Loss: 0.2573006749153137 | Test loss: 0.4198817312717438
    OrderedDict([('weights', tensor([0.3583])), ('bias', tensor([0.1842]))])
    Epoch: 20 | Loss: 0.1632838398218155 | Test loss: 0.30747395753860474
    OrderedDict([('weights', tensor([0.3908])), ('bias', tensor([0.2677]))])
    Epoch: 30 | Loss: 0.07037373632192612 | Test loss: 0.18448665738105774
    OrderedDict([('weights', tensor([0.4274])), ('bias', tensor([0.3581]))])
    Epoch: 40 | Loss: 0.049980729818344116 | Test loss: 0.09902867674827576
    OrderedDict([('weights', tensor([0.4599])), ('bias', tensor([0.4147]))])
    Epoch: 50 | Loss: 0.05151936411857605 | Test loss: 0.07666987180709839
    OrderedDict([('weights', tensor([0.4780])), ('bias', tensor([0.4209]))])
    Epoch: 60 | Loss: 0.04486861824989319 | Test loss: 0.090049609541893
    OrderedDict([('weights', tensor([0.4866])), ('bias', tensor([0.3999]))])
    Epoch: 70 | Loss: 0.04079776257276535 | Test loss: 0.09940623492002487
    OrderedDict([('weights', tensor([0.4975])), ('bias', tensor([0.3808]))])
    Epoch: 80 | Loss: 0.03762822598218918 | Test loss: 0.09353423863649368
    OrderedDict([('weights', tensor([0.5136])), ('bias', tensor([0.3723]))])
    Epoch: 90 | Loss: 0.03402160853147507 | Test loss: 0.0805068239569664
    OrderedDict([('weights', tensor([0.5321])), ('bias', tensor([0.3689]))])
    Epoch: 100 | Loss: 0.030590932816267014 | Test loss: 0.06923215091228485
    OrderedDict([('weights', tensor([0.5501])), ('bias', tensor([0.3642]))])
    Epoch: 110 | Loss: 0.027143195271492004 | Test loss: 0.06230000779032707
    OrderedDict([('weights', tensor([0.5668])), ('bias', tensor([0.3562]))])
    Epoch: 120 | Loss: 0.023709513247013092 | Test loss: 0.05520481616258621
    OrderedDict([('weights', tensor([0.5836])), ('bias', tensor([0.3484]))])
    Epoch: 130 | Loss: 0.0202675499022007 | Test loss: 0.046482883393764496
    OrderedDict([('weights', tensor([0.6009])), ('bias', tensor([0.3417]))])
    Epoch: 140 | Loss: 0.016830043867230415 | Test loss: 0.038598399609327316
    OrderedDict([('weights', tensor([0.6180])), ('bias', tensor([0.3344]))])
    Epoch: 150 | Loss: 0.013395821675658226 | Test loss: 0.03053610399365425
    OrderedDict([('weights', tensor([0.6351])), ('bias', tensor([0.3273]))])
    Epoch: 160 | Loss: 0.009961405768990517 | Test loss: 0.02251533791422844
    OrderedDict([('weights', tensor([0.6521])), ('bias', tensor([0.3201]))])
    Epoch: 170 | Loss: 0.00652692187577486 | Test loss: 0.014441031031310558
    OrderedDict([('weights', tensor([0.6692])), ('bias', tensor([0.3129]))])
    Epoch: 180 | Loss: 0.003093455685302615 | Test loss: 0.0064244926907122135
    OrderedDict([('weights', tensor([0.6863])), ('bias', tensor([0.3057]))])
    Epoch: 190 | Loss: 0.000955758267082274 | Test loss: 0.0022815645206719637
    OrderedDict([('weights', tensor([0.7032])), ('bias', tensor([0.2995]))])

</div>

</div>

<div class="cell code" execution_count="174"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="O6EZVQi1759Y" outputId="e1ea02bd-efe8-4655-eea3-50e88e951189">

``` python
# Check the training loss_valu and test_loss_values
import numpy as np
np.array(torch.tensor(loss_values).numpy()), test_loss_values
```

<div class="output execute_result" execution_count="174">

    (array([0.31172925, 0.25730067, 0.16328384, 0.07037374, 0.04998073,
            0.05151936, 0.04486862, 0.04079776, 0.03762823, 0.03402161,
            0.03059093, 0.0271432 , 0.02370951, 0.02026755, 0.01683004,
            0.01339582, 0.00996141, 0.00652692, 0.00309346, 0.00095576],
           dtype=float32),
     [tensor(0.4906),
      tensor(0.4199),
      tensor(0.3075),
      tensor(0.1845),
      tensor(0.0990),
      tensor(0.0767),
      tensor(0.0900),
      tensor(0.0994),
      tensor(0.0935),
      tensor(0.0805),
      tensor(0.0692),
      tensor(0.0623),
      tensor(0.0552),
      tensor(0.0465),
      tensor(0.0386),
      tensor(0.0305),
      tensor(0.0225),
      tensor(0.0144),
      tensor(0.0064),
      tensor(0.0023)])

</div>

</div>

<div class="cell markdown">

#### 1.5.1.1. <a id='toc1_5_1_1_'></a>[Plot the loss curves](#toc0_)

</div>

<div class="cell code" execution_count="175"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:295}"
id="ccr-GEYe7da1" outputId="68b6980f-85ab-4811-cd53-be0d1091ff10">

``` python
# Plot the loss curves
plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")  # Train loss
plt.plot(epoch_count, test_loss_values, label="Test loss")  # Test loss
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend();
```

<div class="output display_data">

![](efc5fafb1f15c61e73350240d7c144c8ca239d69.png)

</div>

</div>

<div class="cell code" execution_count="176" id="vfl9oAt_09Fd">

``` python
with torch.inference_mode():
  y_preds_new = model_0(X_test)
```

</div>

<div class="cell code" execution_count="177"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="i3Y-ilS3jCdG" outputId="4bc413d2-3b5b-4f4d-c822-5f3693f3796d">

``` python
# Make pred value close to true value
model_0.state_dict()
```

<div class="output execute_result" execution_count="177">

    OrderedDict([('weights', tensor([0.7055])), ('bias', tensor([0.2978]))])

</div>

</div>

<div class="cell code" execution_count="178"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="cE8uVRUeg39p" outputId="9abe949e-3ba1-4d7e-91ee-f0dc59371641">

``` python
# This is true value
weight, bias
```

<div class="output execute_result" execution_count="178">

    (0.7, 0.3)

</div>

</div>

<div class="cell markdown">

### 1.5.2. <a id='toc1_5_2_'></a>[Plot prediction to compare between prediction and Training data](#toc0_)

</div>

<div class="cell code" execution_count="179"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="f_gboBMSl13p" outputId="3c5bce1a-ad41-44ac-bf7f-326d4b5c4308">

``` python
plot_predictions(predictions=y_preds);
```

<div class="output display_data">

![](6a7e97f269487c1df25b142871dcd4a95d6528cc.png)

</div>

</div>

<div class="cell code" execution_count="180"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="9y-u_rVC16XJ" outputId="354933d7-7c14-46c5-9a71-eee4cb7200c2">

``` python
plot_predictions(predictions=y_preds_new);
```

<div class="output display_data">

![](d3c16bdf87721bbe0ac6d451a89b975cac88bde8.png)

</div>

</div>

<div class="cell markdown">

## 1.6. <a id='toc1_6_'></a>[Saving a model in PyTorch](#toc0_)

There are three main methods you should about for saving and loading
models in PyTorch.

1.  `torch.save()` - allows you save a PyTorch object in Python's pickle
    format - <https://docs.python.org/3/library/pickle.html>
2.  `torch.load()` - allows you load a saved PyTorch object
3.  `torch.nn.Module.load_state_dict()` - this allows to load a model's
    saved state dictionary

PyTorch save & load code tutorial + extra-curriculum -
<https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference>

</div>

<div class="cell code" execution_count="181"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="1a0iaBiX5JAG" outputId="dbe254f6-1696-4fa8-a3d3-619aebf930aa">

``` python
# =======0. Saving our PyTorch model=======
from pathlib import Path

# =======1. Create models directory =======
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create subdirectories
state_dict_path = MODEL_PATH / "state_dict"
entire_model_path = MODEL_PATH / "entire_model"
state_dict_path.mkdir(parents=True, exist_ok=True)
entire_model_path.mkdir(parents=True, exist_ok=True)

# =======2. Create model save path=======
# A common PyTorch convention is to save models using either a .pt or .pth file extension
MODEL_NAME = "state_dict/01_pytorch_workflow_model_0_state_dict.pth"  # you also can use .pt file extension
MODEL_NAME_ENTIRE= "entire_model/01_pytorch_workflow_model_0.pth"  # # you also can use .pt file extension
MODEL_STATE_DICT_SAVE_PATH = MODEL_PATH / MODEL_NAME
MODEL_ENTIRE_SAVE_PATH = MODEL_PATH / MODEL_NAME_ENTIRE


# =======3. Save the model state dict=======
# Save state_dict()
print(f"Saving state_dict model to: {MODEL_STATE_DICT_SAVE_PATH}")
torch.save(obj=model_0.state_dict(),
           f=MODEL_STATE_DICT_SAVE_PATH)

# Save entire PyTorch model
print(f"Saving entire model to: {MODEL_ENTIRE_SAVE_PATH}")
torch.save(obj=model_0,
           f = MODEL_ENTIRE_SAVE_PATH)
```

<div class="output stream stdout">

    Saving state_dict model to: models\state_dict\01_pytorch_workflow_model_0_state_dict.pth
    Saving entire model to: models\entire_model\01_pytorch_workflow_model_0.pth

</div>

</div>

<div class="cell code" execution_count="182"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Os6BGzXT54Xq" outputId="2529729e-a3c5-486a-e913-b3031a5fe9ab">

``` python
# =========Check our model if download success=========
list_directory_tree_structure(startpath="models",show_file=True)
```

<div class="output stream stdout">

    models/
        entire_model/
            01_pytorch_workflow_model_0.pth
        state_dict/
            01_pytorch_workflow_model_0_state_dict.pth
            01_pytorch_workflow_model_1_state_dict.pth

</div>

</div>

<div class="cell markdown">

## 1.7. <a id='toc1_7_'></a>[Loading a PyTorch model](#toc0_)

Since we saved our model's `state_dict()` rather the entire model, we'll
create a new instance of our model class and load the saved
`state_dict()` into that.

</div>

<div class="cell code" execution_count="183"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="9U-uXVaC85PP" outputId="a2c8ea18-fbb5-49a7-d985-eab4a76b2875">

``` python
model_0.state_dict()
```

<div class="output execute_result" execution_count="183">

    OrderedDict([('weights', tensor([0.7055])), ('bias', tensor([0.2978]))])

</div>

</div>

<div class="cell code" execution_count="184"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="lTghUxOH89lz" outputId="354c3ac5-96d5-499b-d734-d3e56b8ba3e7">

``` python
# To load in a saved state_dict we have to instantiate a new instance of our model class
loaded_model_0 = LinearRegressionModel()

# Load the saved state_dict of model_0 (this will update the new instance with updated parameters)
loaded_model_0.load_state_dict(torch.load(f=MODEL_STATE_DICT_SAVE_PATH))
```

<div class="output execute_result" execution_count="184">

    <All keys matched successfully>

</div>

</div>

<div class="cell code" execution_count="185"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="_44x4te89OjW" outputId="a8537d36-f89a-4409-90b3-3563f836231e">

``` python
loaded_model_0.state_dict()
```

<div class="output execute_result" execution_count="185">

    OrderedDict([('weights', tensor([0.7055])), ('bias', tensor([0.2978]))])

</div>

</div>

<div class="cell code" execution_count="186"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="BxPDLIU09Q0i" outputId="887f395e-1719-4f68-f3d9-23e17088eb14">

``` python
# Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
  loaded_model_preds = loaded_model_0(X_test)

loaded_model_preds
```

<div class="output execute_result" execution_count="186">

    tensor([[0.8622],
            [0.8763],
            [0.8904],
            [0.9045],
            [0.9186],
            [0.9328],
            [0.9469],
            [0.9610],
            [0.9751],
            [0.9892]])

</div>

</div>

<div class="cell code" execution_count="187"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="3BP9cufq99K-" outputId="191daad3-6df1-4018-95c9-de755eb4f381">

``` python
# Make some models preds
model_0.eval()
with torch.inference_mode():
  y_preds = model_0(X_test)

y_preds
```

<div class="output execute_result" execution_count="187">

    tensor([[0.8622],
            [0.8763],
            [0.8904],
            [0.9045],
            [0.9186],
            [0.9328],
            [0.9469],
            [0.9610],
            [0.9751],
            [0.9892]])

</div>

</div>

<div class="cell code" execution_count="188"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="sSbACbvI94XX" outputId="853b8d5f-7830-41c2-ffd2-93a7fc0b2270">

``` python
# Compare loaded model preds with original model preds
y_preds == loaded_model_preds
```

<div class="output execute_result" execution_count="188">

    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]])

</div>

</div>

<div class="cell code" execution_count="189">

``` python
# Load entire model
loaded_model_0 = torch.load(MODEL_ENTIRE_SAVE_PATH)
loaded_model_0
```

<div class="output execute_result" execution_count="189">

    LinearRegressionModel()

</div>

</div>

<div class="cell code" execution_count="190">

``` python
# Make some predictions with our loaded model
loaded_model_0.eval()
with torch.inference_mode():
  loaded_model_preds = loaded_model_0(X_test)

loaded_model_preds
```

<div class="output execute_result" execution_count="190">

    tensor([[0.8622],
            [0.8763],
            [0.8904],
            [0.9045],
            [0.9186],
            [0.9328],
            [0.9469],
            [0.9610],
            [0.9751],
            [0.9892]])

</div>

</div>

<div class="cell code" execution_count="191">

``` python
# Compare loaded model preds with original model preds
y_preds == loaded_model_preds
```

<div class="output execute_result" execution_count="191">

    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]])

</div>

</div>

<div class="cell markdown">

## 1.8. <a id='toc1_8_'></a>[Putting it all together](#toc0_)

Let's go back through the steps above and see it all in one place.

</div>

<div class="cell code" execution_count="192"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:35}"
id="TY16oebx_4yK" outputId="d0d09199-ab74-45b0-fa8f-ee86ecaa5f3b">

``` python
# Import PyTorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__
```

<div class="output execute_result" execution_count="192">

    '2.0.1+cu118'

</div>

</div>

<div class="cell markdown" id="l91cJBlNAZ7m">

Create device-agnostic code.

This means if we've got access to a GPU, our code will use it (for
potentially faster computing).

If no GPU is available, the code will default to using CPU.

</div>

<div class="cell code" execution_count="193"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="0hRCrpBhAj9G" outputId="d1f41115-e110-49ba-822b-748f48d86bd3">

``` python
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

<div class="output stream stdout">

    Using device: cuda

</div>

</div>

<div class="cell markdown">

### 1.8.1. <a id='toc1_8_1_'></a>[Create Data](#toc0_)

</div>

<div class="cell code" execution_count="194">

``` python
# Create some data using the linear regression formula of y = weight * X + bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (feature and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will pop up
y = bias + weight * X
X[:10], y[:10]
```

<div class="output execute_result" execution_count="194">

    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800],
             [0.1000],
             [0.1200],
             [0.1400],
             [0.1600],
             [0.1800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560],
             [0.3700],
             [0.3840],
             [0.3980],
             [0.4120],
             [0.4260]]))

</div>

</div>

<div class="cell code" execution_count="195">

``` python
# Split data
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(y_train), len(X_test), len(y_test)
```

<div class="output execute_result" execution_count="195">

    (40, 40, 10, 10)

</div>

</div>

<div class="cell code" execution_count="196" id="T4tidX_5CnVx">

``` python
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  # Are there predictions?
  if predictions is not None:
    # Plot the predictions if they exist
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
  
  # Show the legend
  plt.legend(prop={"size": 14});
```

</div>

<div class="cell code" execution_count="197"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="2go898QeCXJY" outputId="ca6406c0-6604-4ff3-8dd4-25aae5a44333">

``` python
# Plot the data
# Note: if you don't have the plot_predictions() function loaded, this will error
plot_predictions(X_train, y_train, X_test, y_test)
```

<div class="output display_data">

![](7f448feaa430fa2daa35c9613fed06c7de4504c2.png)

</div>

</div>

<div class="cell markdown">

### 1.8.2. <a id='toc1_8_2_'></a>[Building a PyTorch Linear model](#toc0_)

</div>

<div class="cell code" execution_count="198">

``` python
# Create a linear model by subclassing nn.Module
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters / also called: linear trasform, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1, # input_shape = 1
                                      out_features=1)   # output_shape = 2
    
    def forward(self, x:torch.Tensor) -> torch.Tensor: # x is input data
        return self.linear_layer(x) # linear layer will do the `weight * X + bias` behind the scence

# Set the manual seed
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
print("model_1 = ", model_1)
print("model_1.state_dict() = ",model_1.state_dict())
```

<div class="output stream stdout">

    model_1 =  LinearRegressionModelV2(
      (linear_layer): Linear(in_features=1, out_features=1, bias=True)
    )
    model_1.state_dict() =  OrderedDict([('linear_layer.weight', tensor([[0.7645]])), ('linear_layer.bias', tensor([0.8300]))])

</div>

</div>

<div class="cell code" execution_count="199"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="v9DmOPoTFIC4" outputId="4f37c379-8cb8-4080-9615-f2d936abd829">

``` python
model_1.state_dict()
```

<div class="output execute_result" execution_count="199">

    OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
                 ('linear_layer.bias', tensor([0.8300]))])

</div>

</div>

<div class="cell code" execution_count="200"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="iq9DxUEyEEvX" outputId="66921f3d-c706-40dc-99ed-6b35f7ec92dc">

``` python
X_train[:5], y_train[:5]
```

<div class="output execute_result" execution_count="200">

    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560]]))

</div>

</div>

<div class="cell markdown">

#### 1.8.2.1. <a id='toc1_8_2_1_'></a>[Let the model use GPU](#toc0_)

</div>

<div class="cell code" execution_count="201"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="x3gCbh6Am8go" outputId="f2121256-84b6-4908-d3bd-084ac636a156">

``` python
# Check the model current device
next(model_1.parameters()).device
```

<div class="output execute_result" execution_count="201">

    device(type='cpu')

</div>

</div>

<div class="cell code" execution_count="202">

``` python
model_1.state_dict()
```

<div class="output execute_result" execution_count="202">

    OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
                 ('linear_layer.bias', tensor([0.8300]))])

</div>

</div>

<div class="cell code" execution_count="203"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="iMNMpAavEFnC" outputId="d3aa990d-455f-4808-fc01-b229be4eb853">

``` python
# Set the model to use the target device
model_1.to(device)
next(model_1.parameters()).device
```

<div class="output execute_result" execution_count="203">

    device(type='cuda', index=0)

</div>

</div>

<div class="cell code" execution_count="204"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="9h7xgGbCnWI1" outputId="74a3d084-6c7b-443b-e217-401985507575">

``` python
model_1.state_dict() 
```

<div class="output execute_result" execution_count="204">

    OrderedDict([('linear_layer.weight', tensor([[0.7645]], device='cuda:0')),
                 ('linear_layer.bias', tensor([0.8300], device='cuda:0'))])

</div>

</div>

<div class="cell markdown">

### 1.8.3. <a id='toc1_8_3_'></a>[Training](#toc0_)

For training we need:

-   Loss function
-   Optimizer
-   Training loop
-   Testing loop

</div>

<div class="cell markdown">

#### 1.8.3.1. <a id='toc1_8_3_1_'></a>[Write Loss function and Optimizer](#toc0_)

</div>

<div class="cell code" execution_count="205" id="BjW4zUvtnOrj">

``` python
# Setup loss function
loss_fn = nn.L1Loss() # same as MAE

# Setup our optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), 
                            lr=0.01)
```

</div>

<div class="cell markdown">

#### 1.8.3.2. <a id='toc1_8_3_2_'></a>[Write Training loop and Testing loop](#toc0_)

</div>

<div class="cell code" execution_count="206"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="GyAumLw3n2Hy" outputId="207e60c6-83b6-4524-da0b-ddf7d7db52e2">

``` python
# Let's write a training loop
torch.manual_seed(42)

epochs = 200

# Put data on the target device (device agnostic code for data) 
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):

  model_1.train()

  # ======Testing======
  # 1. Forward pass
  y_pred = model_1(X_train)

  # 2. Calculate the loss
  loss = loss_fn(y_pred, y_train)

  # 3. Optimizer zero grad
  optimizer.zero_grad()

  # 4. Perform backpropagation
  loss.backward()

  # 5. Optimizer step
  optimizer.step()

  # ======Testing======
  model_1.eval()
  with torch.inference_mode():
    test_pred = model_1(X_test)

    test_loss = loss_fn(test_pred, y_test)

  # Print out what's happening
  if epoch % 10 == 0: 
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
```

<div class="output stream stdout">

    Epoch: 0 | Loss: 0.5551779866218567 | Test loss: 0.5739762187004089
    Epoch: 10 | Loss: 0.439968079328537 | Test loss: 0.4392664134502411
    Epoch: 20 | Loss: 0.3247582018375397 | Test loss: 0.30455657839775085
    Epoch: 30 | Loss: 0.20954833924770355 | Test loss: 0.16984669864177704
    Epoch: 40 | Loss: 0.09433845430612564 | Test loss: 0.03513690456748009
    Epoch: 50 | Loss: 0.023886388167738914 | Test loss: 0.04784907028079033
    Epoch: 60 | Loss: 0.019956795498728752 | Test loss: 0.045803118497133255
    Epoch: 70 | Loss: 0.016517987474799156 | Test loss: 0.037530567497015
    Epoch: 80 | Loss: 0.013089174404740334 | Test loss: 0.02994490973651409
    Epoch: 90 | Loss: 0.009653178043663502 | Test loss: 0.02167237363755703
    Epoch: 100 | Loss: 0.006215683650225401 | Test loss: 0.014086711220443249
    Epoch: 110 | Loss: 0.00278724217787385 | Test loss: 0.005814164876937866
    Epoch: 120 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 130 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 140 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 150 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 160 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 170 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 180 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 190 | Loss: 0.0012645035749301314 | Test loss: 0.013801801018416882

</div>

</div>

<div class="cell code" execution_count="207"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Q2qJjO4ko6x_" outputId="7125869b-b8f4-4ee2-ab14-571afa54c316">

``` python
model_1.state_dict()
```

<div class="output execute_result" execution_count="207">

    OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
                 ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])

</div>

</div>

<div class="cell code" execution_count="208"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="28o1G0gnpYRj" outputId="ae9098ca-52bb-440f-84f6-7979f2ccd2c5">

``` python
# you can observe that the trained weights and biases are very similar to the original weights and biases.
weight, bias 
```

<div class="output execute_result" execution_count="208">

    (0.7, 0.3)

</div>

</div>

<div class="cell markdown">

### 1.8.4. <a id='toc1_8_4_'></a>[Making and evaluating predictions](#toc0_)

</div>

<div class="cell code" execution_count="209"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Ngw4JbJQqubf" outputId="6a05ad5c-98bd-4e01-e951-a40bf84e46a0">

``` python
# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
  y_preds = model_1(X_test)
y_preds
```

<div class="output execute_result" execution_count="209">

    tensor([[0.8600],
            [0.8739],
            [0.8878],
            [0.9018],
            [0.9157],
            [0.9296],
            [0.9436],
            [0.9575],
            [0.9714],
            [0.9854]], device='cuda:0')

</div>

</div>

<div class="cell markdown">

### 1.8.5. <a id='toc1_8_5_'></a>[Plot prediction to compare between prediction and Training data](#toc0_)

</div>

<div class="cell code" execution_count="210"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:428}"
id="uUIbkzHIq5U8" outputId="d1e6ead5-4ce9-42d9-c2a7-332e143408be">

``` python
# Check out our model predictions visually
plot_predictions(predictions=y_preds.cpu()) # If without cpu, with error
```

<div class="output display_data">

![](b616a81cbeb0124f34cfb820345dc604b000bc5e.png)

</div>

</div>

<div class="cell markdown">

### 1.8.6. <a id='toc1_8_6_'></a>[Saving & loading a trained model](#toc0_)

</div>

<div class="cell code" execution_count="211"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="YkPtfsseriP_" outputId="5cc74c45-2adc-43d5-806c-b1446c199eee">

``` python
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models/state_dict")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "01_pytorch_workflow_model_1_state_dict.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(),
           f=MODEL_SAVE_PATH) 
```

<div class="output stream stdout">

    Saving model to: models\state_dict\01_pytorch_workflow_model_1_state_dict.pth

</div>

</div>

<div class="cell code" execution_count="212"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="G5n_xTjssNZ2" outputId="14c63c26-3ae1-458c-d172-2ba03e99479d">

``` python
model_1.state_dict()
```

<div class="output execute_result" execution_count="212">

    OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
                 ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])

</div>

</div>

<div class="cell code" execution_count="213"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="yj8GttgnsNXq" outputId="d934cb26-e89a-4344-f904-26d6cb06487c">

``` python
# Load a PyTorch model

# Create a new instance of lienar regression model V2
loaded_model_1 = LinearRegressionModelV2()

# Load the saved model_1 state_dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put the loaded model to device
loaded_model_1.to(device)
```

<div class="output execute_result" execution_count="213">

    LinearRegressionModelV2(
      (linear_layer): Linear(in_features=1, out_features=1, bias=True)
    )

</div>

</div>

<div class="cell code" execution_count="214"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="4pSqRcShsNVE" outputId="fde1b8e2-cab1-45e8-e8d4-2b2255a2b9d7">

``` python
next(loaded_model_1.parameters()).device
```

<div class="output execute_result" execution_count="214">

    device(type='cuda', index=0)

</div>

</div>

<div class="cell code" execution_count="215"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="S1nhEK4FsNSy" outputId="85d3f5b0-52b8-444f-c1cf-f2d2752cc78c">

``` python
loaded_model_1.state_dict()
```

<div class="output execute_result" execution_count="215">

    OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
                 ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])

</div>

</div>

<div class="cell code" execution_count="216"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="aOZBAa-JsNQJ" outputId="9b722c03-be13-44a8-934d-33613490bb4d">

``` python
# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
  loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds
```

<div class="output execute_result" execution_count="216">

    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]], device='cuda:0')

</div>

</div>

<div class="cell markdown">

## 1.9. <a id='toc1_9_'></a>[Exercises & Extra-curriculum](#toc0_)

For exercise & extra-curriculum, refer to:
<https://www.learnpytorch.io/01_pytorch_workflow/#exercises>

</div>

<div class="cell code" id="5DNZm0YkvEWZ">

``` python
```

</div>
