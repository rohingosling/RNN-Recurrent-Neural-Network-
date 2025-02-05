#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Program: Recurrent Neural Network Test Program
# Version: 1.2
# Release: 2025-01-30
# Author : Rohin Gosling
#
# Description:
# - Test program to verify a basic RNN (Recurrent Neural Network) implementation.
#
# Notes:
#
# - Refresh VS-Code Editor: [Ctrl] + [Shift] + [P]
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn          as nn
import numpy             as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Constants.
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

# Input and training sequence configuration.

SEQUENCE_DELAY  = 16
SEQUENCE_LENGTH = 64                            # Total base sequence length; note that training sequences will add an extra delay.

# Neural network architecture parameters.

NETWORK_INPUT_SIZE  = 1                         # Each time step has 1 feature (binary value).
NETWORK_HIDDEN_SIZE = SEQUENCE_LENGTH           # Hidden size is set equal to the sequence length (design choice for this test).
NETWORK_OUTPUT_SIZE = 1                         # Each time step outputs 1 value.

# Neural network training metaparameters. 

TRAINING_LEARNING_RATE               = 0.001    # Learning rate for the optimizer.
TRAINING_EPOCHS_MAX                  = 200      # Maximum number of epochs to train.
TRAINING_BATCH_SIZE                  = 16       # Number of sequences per batch.
TRAINING_STEPS_PER_EPOCH             = 50       # Number of training steps (batches) per epoch.
TRAINING_EPOCH_PRINT_INTERVAL        = 10       # Interval (in epochs) to print training progress.
TRAINING_GRADIENT_CLIPPING_THRESHOLD = 1.0      # Maximum allowed norm for gradients (to prevent exploding gradients).
TRAINING_VALIDATION_LOSS_THRESHOLD   = 0.0001   # Early stopping threshold for validation loss.

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Implements a basic RNN model for sequence modeling tasks.
#
# Description:
#
# - A simple RNN model using PyTorch's native RNN layer.
# - The network takes a sequence of inputs and returns a sequence of outputs.
# - A final fully connected (linear) layer is applied to each RNN output to produce the final prediction.
#
# Notes:
#
# - The RNN layer processes sequences step-by-step, and its internal hidden state captures past information.
# - The tanh activation function is used to help maintain gradient flow.
# - This model is ideal for learning tasks such as the delayed copy task.
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
class ClassicalRNN ( nn.Module ):
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Constructor.
    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def __init__ ( self, input_size = NETWORK_INPUT_SIZE, hidden_size = NETWORK_HIDDEN_SIZE, output_size = NETWORK_OUTPUT_SIZE, learning_rate = TRAINING_LEARNING_RATE ):
    
        # Initialize the parent class (nn.Module) which is required for all PyTorch models.
    
        super ().__init__ ()
        
        # Create an RNN layer:
        
        self.rnn = nn.RNN (
            input_size   = input_size,    # Number of features per time step.
            hidden_size  = hidden_size,   # Number of neurons in the RNN hidden state.
            batch_first  = True,          # Specify that the batch size is the first dimension. Tensor shape is ( batch_size, seq_len, features ).
            nonlinearity = 'tanh'         # Use tanh activation function for the hidden state.
        )
        
        # Create a final fully connected (linear) layer that maps the hidden state at each time step to the desired output dimension.
    
        self.fc = nn.Linear (
            in_features  = hidden_size,   # The RNN hidden state features.
            out_features = output_size    # The output feature per time step.
        )
        
        # Configure loss function and optimizer.
    
        self.criterion = nn.MSELoss ()
        self.optimizer = torch.optim.Adam ( self.parameters (), lr = learning_rate )

    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    # Processes input sequences through the RNN and applies a linear projection.
    #
    # Description:
    #
    # - This function defines the forward pass of the network.
    # - It accepts a batch of sequences and returns a sequence of outputs.
    # - All time steps from the RNN output are passed through a linear layer to match the target dimensions.
    #
    # Arguments:
    #
    # - x (Tensor): Input tensor with shape ( batch_size, seq_len, input_size ).
    #
    # Return Value:
    #
    # - Tensor: Output tensor with shape ( batch_size, seq_len, output_size ).
    #
    #---------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def forward ( self, x ):
    
        # Process the input sequence through the RNN.
        
        rnn_out, _ = self.rnn ( x )
        
        # Map the RNN outputs to the final output space.
        
        return self.fc ( rnn_out )
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Generates binary sequences with delayed target copies for sequence modeling tasks.
    #
    # Description:
    #
    # - Creates synthetic binary input sequences where the target sequence is a delayed copy
    #   of the input sequence. This is a common toy task to test an RNN’s ability to learn
    #   temporal dependencies.
    #
    # - The inputs are random binary tensors (containing 0s and 1s) with shape:
    #   (batch_size, total_length, 1) where total_length = seq_len + delay.
    #
    # - The targets are created as zero-initialized tensors and then the values after the delay
    #   are filled with the inputs from an earlier time step.
    #
    # Arguments:
    #
    # - batch_size (int): Number of sequences in the batch. Default: TRAINING_BATCH_SIZE.
    #
    # - seq_len (int): Base length of the input sequence (before delay is added). Default: SEQUENCE_LENGTH.
    #
    # - delay (int): The number of time steps by which the target lags behind the input. Default: SEQUENCE_DELAY.
    #
    # Return Value:
    #
    # - Tuple (inputs, targets): Both tensors have shape (batch_size, seq_len + delay, 1).
    #
    # Notes:
    #
    # - The total sequence length is seq_len + delay to allow space for the delayed copy.
    #
    # - Ensure that `delay` is less than seq_len so that the target can contain meaningful data. 
    #
    # - The delay in the input sequence is used to test the model's ability to learn and handle temporal dependencies.
    #   - It simulates a situation where the output occurs a few time steps after the input, which is a common scenario in time-series forecasting and sequence 
    #     modeling applications.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def generate_delayed_copy ( self, batch_size = TRAINING_BATCH_SIZE, seq_len = SEQUENCE_LENGTH, delay = SEQUENCE_DELAY ):
    
        total_length  = seq_len + delay
    
        # 1. Generate random binary input sequences and cast them to float for compatibility with the model.
        # 2. Initialize targets with zeros.
        # 3. Copy the input values into the target with the specified delay.
        
        inputs                   = torch.randint ( 0, 2, ( batch_size, total_length, 1 ) ).float ()
        targets                  = torch.zeros_like ( inputs )
        targets [ :, delay:, : ] = inputs [ :, : -delay, : ]
    
        return inputs, targets

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Executes one training iteration on delayed copy task data.
    #
    # Description:
    #
    # - Generates a new batch of training data using the delayed copy mechanism.
    # - Performs a forward pass through the model to compute outputs.
    # - Calculates the loss over the delayed part of the sequence.
    # - Backpropagates the error, applies gradient clipping to prevent exploding gradients, and updates model parameters using the optimizer.
    #
    # Arguments:
    #
    # - delay (int):                   The delay parameter specifying how many time steps the target is lagged. Default: SEQUENCE_DELAY.
    #
    # Return Value:
    #
    # - float: The scalar loss value for this training step.
    #
    # Notes:
    #
    # - Resetting gradients to zero before backpropagation.
    # - Gradient clipping is applied immediately after backpropagation and before the optimizer step.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def training_step ( self, delay = SEQUENCE_DELAY ):
    
        # Generate a batch of synthetic input and target sequences.
        # Forward pass: compute model outputs.
    
        inputs, targets = self.generate_delayed_copy ( delay = delay )    
        outputs         = self ( inputs )
        
        # Calculate the loss only on the part of the sequence where the delay applies.
    
        loss = self.criterion ( outputs [ :, delay : ], targets [ :, delay : ] )
        
        # Reset gradients to zero before backpropagation.
    
        self.optimizer.zero_grad ()
        loss.backward ()
        
        # Apply gradient clipping to prevent exploding gradients.
    
        torch.nn.utils.clip_grad_norm_ ( self.parameters(), TRAINING_GRADIENT_CLIPPING_THRESHOLD )
        
        # Update model parameters.
    
        self.optimizer.step ()
        
        return loss.item ()

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Calculates validation loss on held-out synthetic data without updating parameters.
    #
    # Description:
    #
    # - Sets the model to evaluation mode to disable dropout (if any) and reduce overhead.
    # - Generates validation data using the same delayed copy method as training.
    # - Computes the loss over several batches and returns the average validation loss.
    #
    # Arguments:
    #
    # - delay (int):       The delay parameter used in data generation.
    #
    # Return Value:
    #
    # - float: The average validation loss computed over the validation batches.
    #
    # Notes:
    #
    # - Training mode vs, evaluation mode.
    # - Use a larger batch size for a more stable estimate of validation loss.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def validate ( self, delay = SEQUENCE_DELAY ):
    
        self.eval()
    
        val_loss        = 0
        num_val_batches = 5    
    
        with torch.no_grad ():
    
            for _ in range ( num_val_batches ):
    
                inputs, targets = self.generate_delayed_copy ( batch_size = 64, delay = delay )
                outputs         = self ( inputs )
                val_loss       += self.criterion ( outputs [ :, delay: ], targets [ :, delay: ] ).item ()
    
        # Switch back to training mode.
    
        self.train()
    
        return val_loss / num_val_batches

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Orchestrates the complete training process with multi-epoch/multi-step execution.
    #
    # Description:
    #
    # - Configures the training loop with a fixed number of epochs and steps per epoch.
    # - At each training step, a batch of data is generated, the model is updated,
    #   and the training loss is computed. 
    # - Periodically computes the validation loss and prints progress.
    # - Implements early stopping if the validation loss falls below a threshold.
    #
    # Notes:
    #
    # - An epoch here is defined as a fixed number of steps (batches), rather than iterating over an entire dataset.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def train_model ( self, epochs = TRAINING_EPOCHS_MAX, steps_per_epoch = TRAINING_STEPS_PER_EPOCH, print_interval = TRAINING_EPOCH_PRINT_INTERVAL, delay = SEQUENCE_DELAY, validation_loss_threshold = TRAINING_VALIDATION_LOSS_THRESHOLD ):
    
        print( f"\nTraining...\n" )
    
        # Main training loop.
    
        for epoch in range ( epochs ):
    
            for step in range ( steps_per_epoch ):
    
                # Each training step performs forward and backward passes as well as parameter update.
                
                loss = self.training_step ( delay )
            
            # After an epoch, perform validation.
    
            validation_loss = self.validate ( delay )
            
            if ( epoch % print_interval == 0 ) or ( validation_loss < validation_loss_threshold ):
                print ( f"Epoch {epoch:4} | Validation Loss: {validation_loss:.8f}"  )
            
            # Early stopping: exit training if validation loss is below threshold.
    
            if validation_loss < validation_loss_threshold:
                break
        
        print ( f"\nTraining complete..." )

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Testing and Inference Class.
#
# Description:
#
# - Encapsulates the testing and inference functions for the trained RNN model.
# - Creates an instance of the ClassicalRNN class, configures it, and trains it.
# - Provides a method to test the trained model with a manually defined input sequence and visualize the results.
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

class RNNTester:
    
    def __init__ ( self ):
    
        # Initialize model components.
    
        print( f"\nInitializing model..." )

        self.model = ClassicalRNN ()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Tests the trained model with a manually created input sequence.
    #
    # Description:
    #
    # - Creates a simple test sequence (manually specified) to examine how well the model has learned the delay.
    # - The expected output is the input sequence shifted by the delay period.
    # - Runs the model on this test input and visualizes both the predicted output and the expected delayed output.
    #
    # Arguments:
    #
    # - delay (int): The delay parameter used during training. Default: SEQUENCE_DELAY.
    #
    # Notes:
    #
    # - Visualization uses matplotlib to show the binary input as a bar chart and compares the model's predictions.
    # - This manual test helps in understanding how well the model is capturing temporal dependencies.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def test_model ( self, delay = SEQUENCE_DELAY ):
    
        # Create a delay padding at the beginning and end of the sequence.
        # - We need the delay to see if the trained model is able to remember the sequence after some number of time steps. 
        
        padding_start_padding = [[0]] * delay
        
        # Create a manually defined binary signal for testing purposes.
        # - This can be any user-selected sequence of binary values. 
        # - Pick a sequence that is easy for a human to identify and evaluate in the output visualizations. 
        
        input_signal = [
            [1], [1], [1], [1], [1], [1], [1], [1],
            [0], [0], [0], [0], [0], [0], [0], [0],
            [1], [0], [1], [0], [1], [0], [1], [0],
            [1], [0], [1], [0], [1], [0], [1], [0],
            [1], [1], [1], [1], [0], [0], [0], [0],
            [1], [1], [0], [0], [1], [1], [0], [0],
            [1], [1], [1], [1], [0], [0], [0], [0],
            [1], [1], [0], [0], [1], [1], [0], [0],
        ]

        # Create a sequence of trailing zeros, so that when we test the model by shifting the input data over to the right by `delay` time steps, we don't cut off 
        # tail of the signal sequence. 
        
        sequence_padding_trailing = padding_start_padding
        
        # Concatenate the delay padding, the signal, and a final delay padding.
        
        test_input = torch.tensor ( padding_start_padding + input_signal + sequence_padding_trailing ).float ().unsqueeze ( 0 )
    
        # Set the PyTorch RNN to evaluation mode, and generate model prediction for the test input.
        
        self.model.eval ()
    
        with torch.no_grad ():
            test_output = self.model ( test_input )
    
        # Convert tensors to numpy arrays for plotting.
    
        input_sequence  = test_input.squeeze  ().numpy ()
        output_sequence = test_output.squeeze ().numpy ()
    
        # Create the expected output by shifting the input sequence by the delay.
    
        output_sequence_expected             = np.zeros_like ( input_sequence )
        output_sequence_expected [ delay : ] = input_sequence [ : -delay ]
    
        # Plot the input and output sequences.
            
        self.plot_results ( input_sequence, output_sequence, output_sequence_expected )

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Chart the results of testing the model. 
    #
    # Description:
    #
    # - Plot the input sequence. 
    # - The expected output is the input sequence shifted by the delay period.
    # - Runs the model on this test input and visualizes both the predicted output and the expected delayed output.
    #
    # Arguments:
    #
    # - delay (int): The delay parameter used during training. Default: SEQUENCE_DELAY.
    #
    # Notes:
    #
    # - Visualization uses matplotlib to show the binary input as a bar chart and compares the model's predictions.
    # - This manual test helps in understanding how well the model is capturing temporal dependencies.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def plot_results ( self, input_sequence, output_sequence, output_sequence_expected ):
        
        plt.figure ( figsize = ( 12, 6 ) )
    
        # Plot the input sequence.
        
        plt.subplot ( 2, 1, 1 )
        plt.bar     ( range ( len ( input_sequence ) ), input_sequence, color = 'black' )
        plt.title   ( 'Input Sequence' )
        plt.xlim    ( 0, len ( input_sequence ) )
    
        # Plot the expected and predicted output.
    
        plt.subplot ( 2, 1, 2 )
        plt.bar     ( range( len ( output_sequence_expected ) ), output_sequence_expected, color = 'gray', label = 'Expected'  )
        plt.scatter ( range( len ( output_sequence ) ),          output_sequence,          color = 'red',  label = 'Predicted' )
        plt.title   ( 'Output Sequence' )
        plt.xlim    ( 0, len ( output_sequence ) )
        plt.legend  ()
    
        plt.tight_layout ()
        plt.show ()
    
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Runs the complete process: training the model and testing it.
    #
    # Description:
    #
    # - Calls the model's training process.
    # - After training, it runs the test sequence to visualize predictions.
    #
    #-------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    def run ( self ):
    
        # Train the model.
    
        self.model.train_model ()
    
        # Test the model.
    
        print ( f"\nTest model..." )
    
        self.test_model()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main function.
#
# Description:
#
# - Creates an instance of the testing and inference class and executes its run method.
#
#-------------------------------------------------------------------------------------------------------------------------------------------------------------

def main ():
    
    rnn_test_function = RNNTester ()
    rnn_test_function.run ()

if __name__ == "__main__":
    main()
