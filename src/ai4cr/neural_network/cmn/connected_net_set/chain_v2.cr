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


ne = Ai4cr::NeuralNetwork::Cmn::MiniNet::ExpV2.new(foo: 1, bar: 7)
nr = Ai4cr::NeuralNetwork::Cmn::MiniNet::ReluV2.new(foo: 2, bar: 8)
nt = Ai4cr::NeuralNetwork::Cmn::MiniNet::TanhV2.new(foo: 3, bar: 9)

arr = [ne,nr,nt]
cv2 = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::ChainV2.new(net_set: arr)
puts "*"*8
puts "cv2: #{cv2.pretty_inspect}"
puts "*"*8

