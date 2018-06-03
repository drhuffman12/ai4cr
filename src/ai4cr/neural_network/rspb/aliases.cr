require "./concerns/*"

# Ai4cr::NeuralNetwork::Rspb::Concerns::InsSet
module Ai4cr
  module NeuralNetwork
    module Rspb
      alias InsSet = Array(Concerns::Outs) # i.e.: Node A's InsSet is made up of one or more other node's Outs
      # alias InsSet = Array(Ins)
      alias IoDataSet = Array(Float64)
    end
  end
end
