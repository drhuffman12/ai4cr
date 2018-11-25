require "./common"
require "./../node/hidden"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Channel
        class Combo < Common(Node::Hidden) # Ai4cr::NeuralNetwork::Rnn::Channel::Combo
          # def node_class
          #   Node::Hidden
          # end

          # def init_nodes
          #   @nodes = time_column_range.map{|t| node_class.new }
          # end
        end
      end
    end
  end
end
      