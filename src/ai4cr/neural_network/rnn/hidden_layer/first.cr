require "./common"
require "./../node_set/*"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module HiddenLayer
        class First < HiddenLayer::Common(NodeSet::Input, NodeSet::Hidden, Channel::Input) # , Channel::Local) # Ai4cr::NeuralNetwork::Rnn::HiddenLayer::First

          # def initialize()
          #   super

          #   # Ai4cr::NeuralNetwork::Rnn::WeightSet::LocalFirst
          # end
        end
      end
    end
  end
end
