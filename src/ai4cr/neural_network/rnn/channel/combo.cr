require "./common"
require "./../node_set/hidden"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Channel
        class Combo < Common(NodeSet::Hidden) # Ai4cr::NeuralNetwork::Rnn::Channel::Combo
          # def node_class
          #   NodeSet::Hidden
          # end

          # def init_nodes
          #   @node_sets = time_column_range.map{|t| node_class.new }
          # end
        end
      end
    end
  end
end
      