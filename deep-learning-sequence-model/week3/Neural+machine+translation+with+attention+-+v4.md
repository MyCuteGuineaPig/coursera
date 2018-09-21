
# Neural Machine Translation

Welcome to your first programming assignment for this week! 

You will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). You will do this using an attention model, one of the most sophisticated sequence to sequence models. 

This notebook was produced together with NVIDIA's Deep Learning Institute. 

Let's load all the packages you will need for this assignment.


```python
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline
```

## 1 - Translating human readable dates into machine readable dates

The model you will build here could be used to translate from one language to another, such as translating from English to Hindi. However, language translation requires massive datasets and usually takes days of training on GPUs. To give you a place to experiment with these models even without using massive datasets, we will instead use a simpler "date translation" task. 

The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) and translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 



<!-- 
Take a look at [nmt_utils.py](./nmt_utils.py) to see all the formatting. Count and figure out how the formats work, you will need this knowledge later. !--> 

### 1.1 - Dataset

We will train the model on a dataset of 10000 human readable dates and their equivalent, standardized, machine readable dates. Let's run the following cells to load the dataset and print some examples. 


```python
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
```

    100%|██████████| 10000/10000 [00:01<00:00, 5378.31it/s]



```python
dataset[:10]
```




    [('3 february 2014', '2014-02-03'),
     ('tuesday october 6 2009', '2009-10-06'),
     ('monday june 21 1971', '1971-06-21'),
     ('9/3/74', '1974-09-03'),
     ('sunday august 7 2005', '2005-08-07'),
     ('monday october 31 1977', '1977-10-31'),
     ('10 aug 1981', '1981-08-10'),
     ('11 06 03', '2003-06-11'),
     ('tuesday march 28 2000', '2000-03-28'),
     ('saturday october 17 2009', '2009-10-17')]



You've loaded:
- `dataset`: a list of tuples of (human readable date, machine readable date)
- `human_vocab`: a python dictionary mapping all characters used in the human readable dates to an integer-valued index 
- `machine_vocab`: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with `human_vocab`. 
- `inv_machine_vocab`: the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 

Let's preprocess the data and map the raw text data into the index values. We will also use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we would have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long). 


```python
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
```

    X.shape: (10000, 30)
    Y.shape: (10000, 10)
    Xoh.shape: (10000, 30, 37)
    Yoh.shape: (10000, 10, 11)


You now have:
- `X`: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to $T_x$ values with a special character (< pad >). `X.shape = (m, Tx)`
- `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
- `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
- `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 


Lets also look at some examples of preprocessed training examples. Feel free to play with `index` in the cell below to navigate the dataset and see how source/target dates are preprocessed. 


```python
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
```

    Source date: 3 february 2014
    Target date: 2014-02-03
    
    Source after preprocessing (indices): [ 6  0 18 17 14 28 31 13 28 34  0  5  3  4  7 36 36 36 36 36 36 36 36 36 36
     36 36 36 36 36]
    Target after preprocessing (indices): [3 1 2 5 0 1 3 0 1 4]
    
    Source after preprocessing (one-hot): [[ 0.  0.  0. ...,  0.  0.  0.]
     [ 1.  0.  0. ...,  0.  0.  0.]
     [ 0.  0.  0. ...,  0.  0.  0.]
     ..., 
     [ 0.  0.  0. ...,  0.  0.  1.]
     [ 0.  0.  0. ...,  0.  0.  1.]
     [ 0.  0.  0. ...,  0.  0.  1.]]
    Target after preprocessing (one-hot): [[ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
     [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]]


## 2 - Neural machine translation with attention

If you had to translate a book's paragraph from French to English, you would not read the whole paragraph, then close the book and translate. Even during the translation process, you would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English you are writing down. 

The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 


### 2.1 - Attention mechanism

In this part, you will implement the attention mechanism presented in the lecture videos. Here is a figure to remind you how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$, which are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$). 

<table>
<td> 
<img src="images/attn_model.png" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td> 
</table>
<caption><center> **Figure 1**: Neural machine translation with attention</center></caption>



Here are some properties of the model that you may notice: 

- There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through $T_x$ time steps; the post-attention LSTM goes through $T_y$ time steps. 

- The post-attention LSTM passes $s^{\langle t \rangle}, c^{\langle t \rangle}$ from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations $s^{\langle t\rangle}$. But since we are using an LSTM here, the LSTM has both the output activation $s^{\langle t\rangle}$ and the hidden cell state $c^{\langle t\rangle}$. However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time $t$ does will not take the specific generated $y^{\langle t-1 \rangle}$ as input; it only takes $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 

- We use $a^{\langle t \rangle} = [\overrightarrow{a}^{\langle t \rangle}; \overleftarrow{a}^{\langle t \rangle}]$ to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM. 

- The diagram on the right uses a `RepeatVector` node to copy $s^{\langle t-1 \rangle}$'s value $T_x$ times, and then `Concatenation` to concatenate $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ to compute $e^{\langle t, t'}$, which is then passed through a softmax to compute $\alpha^{\langle t, t' \rangle}$. We'll explain how to use `RepeatVector` and `Concatenation` in Keras below. 

Lets implement this model. You will start by implementing two functions: `one_step_attention()` and `model()`.

**1) `one_step_attention()`**: At step $t$, given all the hidden states of the Bi-LSTM ($[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$) and the previous hidden state of the second LSTM ($s^{<t-1>}$), `one_step_attention()` will compute the attention weights ($[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$) and output the context vector (see Figure  1 (right) for details):
$$context^{<t>} = \sum_{t' = 0}^{T_x} \alpha^{<t,t'>}a^{<t'>}\tag{1}$$ 

Note that we are denoting the attention in this notebook $context^{\langle t \rangle}$. In the lecture videos, the context was denoted $c^{\langle t \rangle}$, but here we are calling it $context^{\langle t \rangle}$ to avoid confusion with the (post-attention) LSTM's internal memory cell variable, which is sometimes also denoted $c^{\langle t \rangle}$. 
  
**2) `model()`**: Implements the entire model. It first runs the input through a Bi-LSTM to get back $[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$. Then, it calls `one_step_attention()` $T_y$ times (`for` loop). At each iteration of this loop, it gives the computed context vector $c^{<t>}$ to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction $\hat{y}^{<t>}$. 



**Exercise**: Implement `one_step_attention()`. The function `model()` will call the layers in `one_step_attention()` $T_y$ using a for-loop, and it is important that all $T_y$ copies have the same weights. I.e., it should not re-initiaiize the weights every time. In other words, all $T_y$ steps should have shared weights. Here's how you can implement layers with shareable weights in Keras:
1. Define the layer objects (as global variables for examples).
2. Call these objects when propagating the input.

We have defined the layers you need as global variables. Please run the following cells to create them. Please check the Keras documentation to make sure you understand what these layers are: [RepeatVector()](https://keras.io/layers/core/#repeatvector), [Concatenate()](https://keras.io/layers/merge/#concatenate), [Dense()](https://keras.io/layers/core/#dense), [Activation()](https://keras.io/layers/core/#activation), [Dot()](https://keras.io/layers/merge/#dot).


```python
# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)
```

Now you can use these layers to implement `one_step_attention()`. In order to propagate a Keras tensor object X through one of these layers, use `layer(X)` (or `layer([X,Y])` if it requires multiple inputs.), e.g. `densor(X)` will propagate X through the `Dense(1)` layer defined above.


```python
# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e) 
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies) 
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###
    
    return context
```

You will be able to check the expected output of `one_step_attention()` after you've coded the `model()` function.

**Exercise**: Implement `model()` as explained in figure 2 and the text above. Again, we have defined global layers that will share weights to be used in `model()`.


```python
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)
```

Now you can use these layers $T_y$ times in a `for` loop to generate the outputs, and their parameters will not be reinitialized. You will have to carry out the following steps: 

1. Propagate the input into a [Bidirectional](https://keras.io/layers/wrappers/#bidirectional) [LSTM](https://keras.io/layers/recurrent/#lstm)
2. Iterate for $t = 0, \dots, T_y-1$: 
    1. Call `one_step_attention()` on $[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$ and $s^{<t-1>}$ to get the context vector $context^{<t>}$.
    2. Give $context^{<t>}$ to the post-attention LSTM cell. Remember pass in the previous hidden-state $s^{\langle t-1\rangle}$ and cell-states $c^{\langle t-1\rangle}$ of this LSTM using `initial_state= [previous hidden state, previous cell state]`. Get back the new hidden state $s^{<t>}$ and the new cell state $c^{<t>}$.
    3. Apply a softmax layer to $s^{<t>}$, get the output. 
    4. Save the output by adding it to the list of outputs.

3. Create your Keras model instance, it should have three inputs ("inputs", $s^{<0>}$ and $c^{<0>}$) and output the list of "outputs".


```python
# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True),input_shape=(m, Tx, n_a*2))(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context,initial_state = [s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs+=[out]
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs = [X,s0,c0],outputs = outputs)
    
    ### END CODE HERE ###
    
    return model
```

Run the following cell to create your model.


```python
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
```

Let's get a summary of the model to check if it matches the expected output.


```python
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_6 (InputLayer)             (None, 30, 37)        0                                            
    ____________________________________________________________________________________________________
    s0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    bidirectional_6 (Bidirectional)  (None, 30, 64)        17920       input_6[0][0]                    
    ____________________________________________________________________________________________________
    repeat_vector_1 (RepeatVector)   (None, 30, 64)        0           s0[0][0]                         
                                                                       lstm_9[10][0]                    
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[18][0]                    
    ____________________________________________________________________________________________________
    concatenate_1 (Concatenate)      (None, 30, 128)       0           bidirectional_6[0][0]            
                                                                       repeat_vector_1[40][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[41][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[42][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[43][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[44][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[45][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[46][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[47][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[48][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[49][0]           
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 30, 10)        1290        concatenate_1[40][0]             
                                                                       concatenate_1[41][0]             
                                                                       concatenate_1[42][0]             
                                                                       concatenate_1[43][0]             
                                                                       concatenate_1[44][0]             
                                                                       concatenate_1[45][0]             
                                                                       concatenate_1[46][0]             
                                                                       concatenate_1[47][0]             
                                                                       concatenate_1[48][0]             
                                                                       concatenate_1[49][0]             
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 30, 1)         11          dense_1[40][0]                   
                                                                       dense_1[41][0]                   
                                                                       dense_1[42][0]                   
                                                                       dense_1[43][0]                   
                                                                       dense_1[44][0]                   
                                                                       dense_1[45][0]                   
                                                                       dense_1[46][0]                   
                                                                       dense_1[47][0]                   
                                                                       dense_1[48][0]                   
                                                                       dense_1[49][0]                   
    ____________________________________________________________________________________________________
    attention_weights (Activation)   multiple              0           dense_2[40][0]                   
                                                                       dense_2[41][0]                   
                                                                       dense_2[42][0]                   
                                                                       dense_2[43][0]                   
                                                                       dense_2[44][0]                   
                                                                       dense_2[45][0]                   
                                                                       dense_2[46][0]                   
                                                                       dense_2[47][0]                   
                                                                       dense_2[48][0]                   
                                                                       dense_2[49][0]                   
    ____________________________________________________________________________________________________
    dot_1 (Dot)                      multiple              0           attention_weights[40][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[41][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[42][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[43][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[44][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[45][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[46][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[47][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[48][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[49][0]         
                                                                       bidirectional_6[0][0]            
    ____________________________________________________________________________________________________
    c0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    lstm_9 (LSTM)                    [(None, 64), (None, 6 33024       dot_1[40][0]                     
                                                                       s0[0][0]                         
                                                                       c0[0][0]                         
                                                                       dot_1[41][0]                     
                                                                       lstm_9[10][0]                    
                                                                       lstm_9[10][2]                    
                                                                       dot_1[42][0]                     
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[11][2]                    
                                                                       dot_1[43][0]                     
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[12][2]                    
                                                                       dot_1[44][0]                     
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[13][2]                    
                                                                       dot_1[45][0]                     
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[14][2]                    
                                                                       dot_1[46][0]                     
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[15][2]                    
                                                                       dot_1[47][0]                     
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[16][2]                    
                                                                       dot_1[48][0]                     
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[17][2]                    
                                                                       dot_1[49][0]                     
                                                                       lstm_9[18][0]                    
                                                                       lstm_9[18][2]                    
    ____________________________________________________________________________________________________
    dense_7 (Dense)                  (None, 11)            715         lstm_9[10][0]                    
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[18][0]                    
                                                                       lstm_9[19][0]                    
    ====================================================================================================
    Total params: 52,960
    Trainable params: 52,960
    Non-trainable params: 0
    ____________________________________________________________________________________________________


**Expected Output**:

Here is the summary you should see
<table>
    <tr>
        <td>
            **Total params:**
        </td>
        <td>
         52,960
        </td>
    </tr>
        <tr>
        <td>
            **Trainable params:**
        </td>
        <td>
         52,960
        </td>
    </tr>
            <tr>
        <td>
            **Non-trainable params:**
        </td>
        <td>
         0
        </td>
    </tr>
                    <tr>
        <td>
            **bidirectional_1's output shape **
        </td>
        <td>
         (None, 30, 64)  
        </td>
    </tr>
    <tr>
        <td>
            **repeat_vector_1's output shape **
        </td>
        <td>
         (None, 30, 64) 
        </td>
    </tr>
                <tr>
        <td>
            **concatenate_1's output shape **
        </td>
        <td>
         (None, 30, 128) 
        </td>
    </tr>
            <tr>
        <td>
            **attention_weights's output shape **
        </td>
        <td>
         (None, 30, 1)  
        </td>
    </tr>
        <tr>
        <td>
            **dot_1's output shape **
        </td>
        <td>
         (None, 1, 64)
        </td>
    </tr>
           <tr>
        <td>
            **dense_3's output shape **
        </td>
        <td>
         (None, 11) 
        </td>
    </tr>
</table>


As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use. Compile your model using `categorical_crossentropy` loss, a custom [Adam](https://keras.io/optimizers/#adam) [optimizer](https://keras.io/optimizers/#usage-of-optimizers) (`learning rate = 0.005`, $\beta_1 = 0.9$, $\beta_2 = 0.999$, `decay = 0.01`)  and `['accuracy']` metrics:


```python
### START CODE HERE ### (≈2 lines)
out = Adam(lr=0.005, beta_1=0.9, beta_2=0.999,decay=0.01) 
model.compile(loss='categorical_crossentropy', optimizer=out,metrics=['accuracy'])
### END CODE HERE ###
```

The last step is to define all your inputs and outputs to fit the model:
- You already have X of shape $(m = 10000, T_x = 30)$ containing the training examples.
- You need to create `s0` and `c0` to initialize your `post_activation_LSTM_cell` with 0s.
- Given the `model()` you coded, you need the "outputs" to be a list of 11 elements of shape (m, T_y). So that: `outputs[i][0], ..., outputs[i][Ty]` represent the true labels (characters) corresponding to the $i^{th}$ training example (`X[i]`). More generally, `outputs[i][j]` is the true label of the $j^{th}$ character in the $i^{th}$ training example.


```python
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
```

Let's now fit the model and run it for one epoch.


```python
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
```

    Epoch 1/1
    10000/10000 [==============================] - 32s - loss: 17.1269 - dense_7_loss_1: 1.4205 - dense_7_loss_2: 1.1065 - dense_7_loss_3: 1.7916 - dense_7_loss_4: 2.7314 - dense_7_loss_5: 0.7983 - dense_7_loss_6: 1.3258 - dense_7_loss_7: 2.6811 - dense_7_loss_8: 0.9769 - dense_7_loss_9: 1.7143 - dense_7_loss_10: 2.5805 - dense_7_acc_1: 0.4050 - dense_7_acc_2: 0.6487 - dense_7_acc_3: 0.2839 - dense_7_acc_4: 0.0674 - dense_7_acc_5: 0.9546 - dense_7_acc_6: 0.2947 - dense_7_acc_7: 0.0613 - dense_7_acc_8: 0.9477 - dense_7_acc_9: 0.2330 - dense_7_acc_10: 0.1037    





    <keras.callbacks.History at 0x7f2d7e057ba8>



While training you can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives you an example of what the accuracies could be if the batch had 2 examples: 

<img src="images/table.png" style="width:700;height:200px;"> <br>
<caption><center>Thus, `dense_2_acc_8: 0.89` means that you are predicting the 7th character of the output correctly 89% of the time in the current batch of data. </center></caption>


We have run this model for longer, and saved the weights. Run the next cell to load our weights. (By training a model for several minutes, you should be able to obtain a model of similar accuracy, but loading our model will save you time.) 


```python
model.load_weights('models/model.h5')
```

You can now see the results on new examples.


```python
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))
```

    source: 3 May 1979
    output: 1979-05-03
    source: 5 April 09
    output: 2009-05-05
    source: 21th of August 2016
    output: 2016-08-21
    source: Tue 10 Jul 2007
    output: 2007-07-10
    source: Saturday May 9 2018
    output: 2018-05-09
    source: March 3 2001
    output: 2001-03-03
    source: March 3rd 2001
    output: 2001-03-03
    source: 1 March 2001
    output: 2001-03-01


You can also change these examples to test with your own examples. The next part will give you a better sense on what the attention mechanism is doing--i.e., what part of the input the network is paying attention to when generating a particular output character. 

## 3 - Visualizing Attention (Optional / Ungraded)

Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (say the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can  visualize what part of the output is looking at what part of the input.

Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed $\alpha^{\langle t, t' \rangle}$ we get this: 

<img src="images/date_attention.png" style="width:600;height:300px;"> <br>
<caption><center> **Figure 8**: Full Attention Map</center></caption>

Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We see also that 9 has been translated as 09 and May has been correctly translated into 05, with the output paying attention to the parts of the input it needs to to make the translation. The year mostly requires it to pay attention to the input's "18" in order to generate "2018." 



### 3.1 - Getting the activations from the network

Lets now visualize the attention values in your network. We'll propagate an example through the network, then visualize the values of $\alpha^{\langle t, t' \rangle}$. 

To figure out where the attention values are located, let's start by printing a summary of the model .


```python
model.summary()
```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_6 (InputLayer)             (None, 30, 37)        0                                            
    ____________________________________________________________________________________________________
    s0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    bidirectional_6 (Bidirectional)  (None, 30, 64)        17920       input_6[0][0]                    
    ____________________________________________________________________________________________________
    repeat_vector_1 (RepeatVector)   (None, 30, 64)        0           s0[0][0]                         
                                                                       lstm_9[10][0]                    
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[18][0]                    
    ____________________________________________________________________________________________________
    concatenate_1 (Concatenate)      (None, 30, 128)       0           bidirectional_6[0][0]            
                                                                       repeat_vector_1[40][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[41][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[42][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[43][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[44][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[45][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[46][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[47][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[48][0]           
                                                                       bidirectional_6[0][0]            
                                                                       repeat_vector_1[49][0]           
    ____________________________________________________________________________________________________
    dense_1 (Dense)                  (None, 30, 10)        1290        concatenate_1[40][0]             
                                                                       concatenate_1[41][0]             
                                                                       concatenate_1[42][0]             
                                                                       concatenate_1[43][0]             
                                                                       concatenate_1[44][0]             
                                                                       concatenate_1[45][0]             
                                                                       concatenate_1[46][0]             
                                                                       concatenate_1[47][0]             
                                                                       concatenate_1[48][0]             
                                                                       concatenate_1[49][0]             
    ____________________________________________________________________________________________________
    dense_2 (Dense)                  (None, 30, 1)         11          dense_1[40][0]                   
                                                                       dense_1[41][0]                   
                                                                       dense_1[42][0]                   
                                                                       dense_1[43][0]                   
                                                                       dense_1[44][0]                   
                                                                       dense_1[45][0]                   
                                                                       dense_1[46][0]                   
                                                                       dense_1[47][0]                   
                                                                       dense_1[48][0]                   
                                                                       dense_1[49][0]                   
    ____________________________________________________________________________________________________
    attention_weights (Activation)   multiple              0           dense_2[40][0]                   
                                                                       dense_2[41][0]                   
                                                                       dense_2[42][0]                   
                                                                       dense_2[43][0]                   
                                                                       dense_2[44][0]                   
                                                                       dense_2[45][0]                   
                                                                       dense_2[46][0]                   
                                                                       dense_2[47][0]                   
                                                                       dense_2[48][0]                   
                                                                       dense_2[49][0]                   
    ____________________________________________________________________________________________________
    dot_1 (Dot)                      multiple              0           attention_weights[40][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[41][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[42][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[43][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[44][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[45][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[46][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[47][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[48][0]         
                                                                       bidirectional_6[0][0]            
                                                                       attention_weights[49][0]         
                                                                       bidirectional_6[0][0]            
    ____________________________________________________________________________________________________
    c0 (InputLayer)                  (None, 64)            0                                            
    ____________________________________________________________________________________________________
    lstm_9 (LSTM)                    [(None, 64), (None, 6 33024       dot_1[40][0]                     
                                                                       s0[0][0]                         
                                                                       c0[0][0]                         
                                                                       dot_1[41][0]                     
                                                                       lstm_9[10][0]                    
                                                                       lstm_9[10][2]                    
                                                                       dot_1[42][0]                     
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[11][2]                    
                                                                       dot_1[43][0]                     
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[12][2]                    
                                                                       dot_1[44][0]                     
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[13][2]                    
                                                                       dot_1[45][0]                     
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[14][2]                    
                                                                       dot_1[46][0]                     
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[15][2]                    
                                                                       dot_1[47][0]                     
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[16][2]                    
                                                                       dot_1[48][0]                     
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[17][2]                    
                                                                       dot_1[49][0]                     
                                                                       lstm_9[18][0]                    
                                                                       lstm_9[18][2]                    
    ____________________________________________________________________________________________________
    dense_7 (Dense)                  (None, 11)            715         lstm_9[10][0]                    
                                                                       lstm_9[11][0]                    
                                                                       lstm_9[12][0]                    
                                                                       lstm_9[13][0]                    
                                                                       lstm_9[14][0]                    
                                                                       lstm_9[15][0]                    
                                                                       lstm_9[16][0]                    
                                                                       lstm_9[17][0]                    
                                                                       lstm_9[18][0]                    
                                                                       lstm_9[19][0]                    
    ====================================================================================================
    Total params: 52,960
    Trainable params: 52,960
    Non-trainable params: 0
    ____________________________________________________________________________________________________


Navigate through the output of `model.summary()` above. You can see that the layer named `attention_weights` outputs the `alphas` of shape (m, 30, 1) before `dot_2` computes the context vector for every time step $t = 0, \ldots, T_y-1$. Lets get the activations from this layer.

The function `attention_map()` pulls out the attention values from your model and plots them.


```python
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1138     try:
    -> 1139       return fn(*args)
       1140     except errors.OpError as e:


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
       1120                                  feed_dict, fetch_list, target_list,
    -> 1121                                  status, run_metadata)
       1122 


    /opt/conda/lib/python3.6/contextlib.py in __exit__(self, type, value, traceback)
         88             try:
    ---> 89                 next(self.gen)
         90             except StopIteration:


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py in raise_exception_on_not_ok_status()
        465           compat.as_text(pywrap_tensorflow.TF_Message(status)),
    --> 466           pywrap_tensorflow.TF_GetCode(status))
        467   finally:


    InvalidArgumentError: Shape [-1,30,37] has negative dimensions
    	 [[Node: input_2 = Placeholder[dtype=DT_FLOAT, shape=[?,30,37], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

    
    During handling of the above exception, another exception occurred:


    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-53-4b75f91f2700> in <module>()
    ----> 1 attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
    

    /home/jovyan/work/Week 3/Machine Translation/nmt_utils.py in plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text, n_s, num, Tx, Ty)
        197 
        198     f = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    --> 199     r = f([encoded, s0, c0])
        200 
        201     for t in range(Ty):


    /opt/conda/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py in __call__(self, inputs)
       2268         updated = session.run(self.outputs + [self.updates_op],
       2269                               feed_dict=feed_dict,
    -> 2270                               **self.session_kwargs)
       2271         return updated[:len(self.outputs)]
       2272 


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in run(self, fetches, feed_dict, options, run_metadata)
        787     try:
        788       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 789                          run_metadata_ptr)
        790       if run_metadata:
        791         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in _run(self, handle, fetches, feed_dict, options, run_metadata)
        995     if final_fetches or final_targets:
        996       results = self._do_run(handle, final_targets, final_fetches,
    --> 997                              feed_dict_string, options, run_metadata)
        998     else:
        999       results = []


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1130     if handle is None:
       1131       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
    -> 1132                            target_list, options, run_metadata)
       1133     else:
       1134       return self._do_call(_prun_fn, self._session, handle, feed_dict,


    /opt/conda/lib/python3.6/site-packages/tensorflow/python/client/session.py in _do_call(self, fn, *args)
       1150         except KeyError:
       1151           pass
    -> 1152       raise type(e)(node_def, op, message)
       1153 
       1154   def _extend_graph(self):


    InvalidArgumentError: Shape [-1,30,37] has negative dimensions
    	 [[Node: input_2 = Placeholder[dtype=DT_FLOAT, shape=[?,30,37], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
    
    Caused by op 'input_2', defined at:
      File "/opt/conda/lib/python3.6/runpy.py", line 193, in _run_module_as_main
        "__main__", mod_spec)
      File "/opt/conda/lib/python3.6/runpy.py", line 85, in _run_code
        exec(code, run_globals)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/__main__.py", line 3, in <module>
        app.launch_new_instance()
      File "/opt/conda/lib/python3.6/site-packages/traitlets/config/application.py", line 658, in launch_instance
        app.start()
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/kernelapp.py", line 474, in start
        ioloop.IOLoop.instance().start()
      File "/opt/conda/lib/python3.6/site-packages/zmq/eventloop/ioloop.py", line 177, in start
        super(ZMQIOLoop, self).start()
      File "/opt/conda/lib/python3.6/site-packages/tornado/ioloop.py", line 887, in start
        handler_func(fd_obj, events)
      File "/opt/conda/lib/python3.6/site-packages/tornado/stack_context.py", line 275, in null_wrapper
        return fn(*args, **kwargs)
      File "/opt/conda/lib/python3.6/site-packages/zmq/eventloop/zmqstream.py", line 440, in _handle_events
        self._handle_recv()
      File "/opt/conda/lib/python3.6/site-packages/zmq/eventloop/zmqstream.py", line 472, in _handle_recv
        self._run_callback(callback, msg)
      File "/opt/conda/lib/python3.6/site-packages/zmq/eventloop/zmqstream.py", line 414, in _run_callback
        callback(*args, **kwargs)
      File "/opt/conda/lib/python3.6/site-packages/tornado/stack_context.py", line 275, in null_wrapper
        return fn(*args, **kwargs)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/kernelbase.py", line 276, in dispatcher
        return self.dispatch_shell(stream, msg)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/kernelbase.py", line 228, in dispatch_shell
        handler(stream, idents, msg)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/kernelbase.py", line 390, in execute_request
        user_expressions, allow_stdin)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/ipkernel.py", line 196, in do_execute
        res = shell.run_cell(code, store_history=store_history, silent=silent)
      File "/opt/conda/lib/python3.6/site-packages/ipykernel/zmqshell.py", line 501, in run_cell
        return super(ZMQInteractiveShell, self).run_cell(*args, **kwargs)
      File "/opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2717, in run_cell
        interactivity=interactivity, compiler=compiler, result=result)
      File "/opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2821, in run_ast_nodes
        if self.run_code(code, result):
      File "/opt/conda/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2881, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "<ipython-input-15-5f649440f586>", line 1, in <module>
        model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
      File "<ipython-input-14-57369432a3d2>", line 19, in model
        X = Input(shape=(Tx, human_vocab_size))
      File "/opt/conda/lib/python3.6/site-packages/keras/engine/topology.py", line 1436, in Input
        input_tensor=tensor)
      File "/opt/conda/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 87, in wrapper
        return func(*args, **kwargs)
      File "/opt/conda/lib/python3.6/site-packages/keras/engine/topology.py", line 1347, in __init__
        name=self.name)
      File "/opt/conda/lib/python3.6/site-packages/keras/backend/tensorflow_backend.py", line 439, in placeholder
        x = tf.placeholder(dtype, shape=shape, name=name)
      File "/opt/conda/lib/python3.6/site-packages/tensorflow/python/ops/array_ops.py", line 1530, in placeholder
        return gen_array_ops._placeholder(dtype=dtype, shape=shape, name=name)
      File "/opt/conda/lib/python3.6/site-packages/tensorflow/python/ops/gen_array_ops.py", line 1954, in _placeholder
        name=name)
      File "/opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/op_def_library.py", line 767, in apply_op
        op_def=op_def)
      File "/opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 2506, in create_op
        original_op=self._default_original_op, op_def=op_def)
      File "/opt/conda/lib/python3.6/site-packages/tensorflow/python/framework/ops.py", line 1269, in __init__
        self._traceback = _extract_stack()
    
    InvalidArgumentError (see above for traceback): Shape [-1,30,37] has negative dimensions
    	 [[Node: input_2 = Placeholder[dtype=DT_FLOAT, shape=[?,30,37], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]



On the generated plot you can observe the values of the attention weights for each character of the predicted output. Examine this plot and check that where the network is paying attention makes sense to you.

In the date translation application, you will observe that most of the time attention helps predict the year, and hasn't much impact on predicting the day/month.

### Congratulations!


You have come to the end of this assignment 

<font color='blue'> **Here's what you should remember from this notebook**:

- Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation. 
- An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
- A network using an attention mechanism can translate from inputs of length $T_x$ to outputs of length $T_y$, where $T_x$ and $T_y$ can be different. 
- You can visualize attention weights $\alpha^{\langle t,t' \rangle}$ to see what the network is paying attention to while generating each output.

Congratulations on finishing this assignment! You are now able to implement an attention model and use it to learn complex mappings from one sequence to another. 


```python

```


```python

```
