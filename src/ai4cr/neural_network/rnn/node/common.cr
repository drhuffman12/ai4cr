# require "./math"
# require "./aliases"
require "./interface"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        class Common # Ai4cr::NeuralNetwork::Rnn::Node::Common
          include Node::Interface
  
          getter state_qty : Int32
          getter state_range : Range(Int32, Int32)
  
          property state_values : Array(Float64)
  
          def initialize(@state_qty = STATE_QTY_DEFAULT) #, @memory_qty = 1)
            # init_states(state_qty, memory_qty)
            @state_range = (0..state_qty-1)
            @state_values = init_state_values
          end
  
          # def init_states
          #   @state_range = (0..state_qty-1)
          #   @state_values = init_state_values
          # end
  
          def init_state_values
            state_range.map{ |s| 0.0 }
          end
        end
      end
    end
  end
end
      