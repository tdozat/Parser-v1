# RNN Cells
The recurrent cells that this model allows for all inherit from BaseCell, which saves information about the number of units in each cell and which includes a trainable initial state. In all cells, bias terms are added to the cell gates but not the cell activations; this is because the initial state effectively acts as the bias term, and adding biases to the cell activation means always expanding/shrinking the cell state over time no matter what the input is, which is clearly undesireable. The possible cells include:

* `RNNCell`: A vanilla RNN cell with no gates.
* `LSTMCell`: A vanilla LSTM cell with three gates; the value of the output gate is computed from input from the lower layer and the previous hidden state, with no input from the current cell state. This is to allow all activations--the recurrent activation and the gate activations--to be computable with one matrix multiplication, making the network more efficient. Conceptually, using the cell state is probably undesireable anyway since the scale of the cell state can get more extreme over time, meaning that in a network that did use cell states to compute the output gate, the value of the output gate would be primarily determined by the input/previous hidden state in the first few steps but later on would be dominated by extreme values in the cell state.
* `CifLSTMCell`: An LSTM cell that uses a coupled input-forget gate:
```python
# Vanilla LSTM cell
candidate_cell_state = tanh(cell_activation)
current_cell_state = input_gate * candidate_cell_state + (1-forget_gate) * previous_cell_state
# Coupled input-forget LSTM cell
current_cell_state = update_gate * cell_activation + (1-update_gate) * previous_cell_state
```

In standard LSTM cells, tanh is applied to the cell activation in order to keep the cell state from exploding; in a CifLSTM cell, the cell state is a weighted average rather than a weighted sum, so it can't explode and the tanh isn't necessary.
* `GRUCell`: a vanilla GRU cell with two gates.
