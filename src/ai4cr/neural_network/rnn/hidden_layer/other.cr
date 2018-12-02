require "./common"
require "./../node_set/*"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module HiddenLayer
        class Other < HiddenLayer::Common(NodeSet::Hidden, NodeSet::Hidden, Channel::Combo, Channel::Combo) # Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other
        end
      end
    end
  end
end
