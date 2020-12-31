require "json"
require "./learning_style.cr"
require "./rnn_concerns/calc_guess.cr"
require "./rnn_concerns/props_and_inits.cr"

# require "./rnn_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class RnnSimple
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

        include JSON::Serializable
        include RnnConcerns::PropsAndInits
        include RnnConcerns::CalcGuess

        # def init_mini_net_set
        #   synaptic_layer_indexes.map do |li|
        #     mn_output_size = node_output_sizes[li]
        #     time_col_indexes.map do |ti|
        #       mn_input_size = node_input_sizes[li][ti].values.sum
        #       MiniNet.new(height: mn_input_size, width: mn_output_size) # TODO: Add more params. (For now, use defaults.)
        #     end
        #   end
        # end
      end
    end
  end
end
