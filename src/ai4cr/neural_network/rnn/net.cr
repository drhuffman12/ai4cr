require "./node_set/*"
require "./channel/*"
require "./hidden_layer/*"

module Ai4cr
  module NeuralNetwork
    module Rnn # aka RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnn::Net 
        getter hidden_layer_qty : Int32
        getter time_column_qty : Int32
        getter memory_layer_qty : Int32

        getter output_state_qty : Int32
        getter hidden_state_qty : Int32
        getter input_state_qty : Int32

        getter dendrite_offsets : Array(Int32)
        getter bias : Bool
        getter output_winner_qty : Int32

        property channel_output : Channel::Output
        property hidden_layers : Array(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::First | Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other)
        property channel_input : Channel::Input
        
        def initialize(
          @hidden_layer_qty = 2, @time_column_qty = 4, @memory_layer_qty = NodeSet::Hidden::MEMORY_QTY_DEFAULT,
          @output_state_qty = 3, @hidden_state_qty = 4, @input_state_qty = 2,
          @dendrite_offsets = Channel::Interface::DENDRITE_OFFSETS_DEFAULT,
          @bias = true,
          @output_winner_qty = 1 # when guessing, exaggerate top n number of output states to maximum; others get minimized
        )
          @channel_output = Channel::Output.new(time_column_qty: time_column_qty, state_qty: output_state_qty)

          @channel_input = Channel::Input.new(time_column_qty: time_column_qty, state_qty: input_state_qty)

          @hidden_layers = Array(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::First | Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other).new
          @hidden_layers << HiddenLayer::First.new(previous_layer_output_channel: channel_input, bias: true, output_winner_qty: output_winner_qty, layer_index: 0, time_column_qty: time_column_qty, dendrite_offsets: dendrite_offsets, state_qty: hidden_state_qty)
          
          (1..hidden_layer_qty-1).map do |h|
            HiddenLayer::Other.new(previous_layer_output_channel: hidden_layers[0].channel_combo, bias: false, output_winner_qty: output_winner_qty, layer_index: h, time_column_qty: time_column_qty, dendrite_offsets: dendrite_offsets, state_qty: hidden_state_qty)
          end
        end

        # Training
        def train # (inputs, outputs)
          guess
          correct
        end

        # Forward Guessing
        def guess # aka eval and feed_forward
          guess_inputs_to_hidden
          (1..hidden_layer_qty-2).each do |hidden_layer_index|
            guess_hidden_to_hidden(hidden_layer_index)
          end
          guess_hidden_to_output
          # exaggerate_top_n_output_states
        end

        def guess_inputs_to_hidden
        end

        def guess_hidden_to_hidden(hidden_layer_index)
        end

        def guess_hidden_to_output
        end

        def exaggerate_top_n_output_states
        end

        # Backward Training
        # Propagate error backwards
        def correct # aka backpropagate # (expected_output_values)
          # check_output_dimension(expected_output_values.size)
          calculate_output_deltas # (expected_output_values)
          calculate_internal_deltas
          update_weights
        end

        def calculate_output_deltas # (expected_output_values)
        end

        def calculate_internal_deltas 
        end

        def update_weights 
        end
      end
    end
  end
end
