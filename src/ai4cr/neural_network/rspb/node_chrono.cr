# Ai4cr::NeuralNetwork::Concerns::Backprop
require "./concerns/outs"

module Ai4cr
  module NeuralNetwork
    module Rspb
      # struct PrevMem
      struct NodeChrono
        include Concerns::Outs

        property in_below, in_chron, in_mem
        property weights
        # property output

        # @in_below : Array(Float64) = [0.0]
        # @in_chron : Array(Float64) = [0.0]
        # @in_mem : Array(Float64) = [0.0]
        # @weights : Array(Array(Float64)) = [[0.0]]
        # @output : Array(Float64) = [0.0]

        def initialize(node_below : Outs, node_chrono : Outs, node_mem : Outs)
        end


        def update
          i = -1
          inputs_set.each do |inputs|
            inputs.each do |in_val|
              i += 1
              @outputs[i] = in_val
            end
          end
        end
      end
    end
  end
end
