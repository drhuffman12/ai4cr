require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module ConnectedNetSet
        class ChainV2
          property net_set : Array(MiniNet::AbstractNet)

          def initialize(@net_set)
          end

        end
      end
    end
  end
end

ne = Ai4cr::NeuralNetwork::Cmn::MiniNet::ExpV2.new(height: 1, width: 7)
nr = Ai4cr::NeuralNetwork::Cmn::MiniNet::ReluV2.new(height: 2, width: 8)
nt = Ai4cr::NeuralNetwork::Cmn::MiniNet::TanhV2.new(height: 3, width: 9)

arr = [ne,nr,nt]
cv2 = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::ChainV2.new(net_set: arr)
puts "*"*8
puts "cv2: #{cv2.pretty_inspect}"
puts "*"*8

