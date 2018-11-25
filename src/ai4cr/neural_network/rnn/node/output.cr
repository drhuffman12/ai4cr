# require "./math"
# require "./aliases"
require "./trainable"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        class Output < Trainable # Ai4cr::NeuralNetwork::Rnn::Node::Output
  
          property state_expectations : Array(Float64)
          property state_exaggerates : Array(Float64)
  
          # memory aka history
          property memory_expectations : Array(Array(Float64))
  
          def initialize(@state_qty = STATE_QTY_DEFAULT, @memory_qty = MEMORY_QTY_DEFAULT)
            super # (state_qty)
            # init_states
            @state_expectations = init_state_values
            @state_exaggerates = init_state_values
            # init_memories
            @memory_expectations = init_memory_values
          end
        end
      end
    end
  end
end
      