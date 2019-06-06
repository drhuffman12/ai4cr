require "json"
# require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct Net
          include JSON::Serializable

          property node_input_mappings

          def initialize(@node_input_mappings : Array(NodeCoord))
          end

          
        end
      end
    end
  end
end
      