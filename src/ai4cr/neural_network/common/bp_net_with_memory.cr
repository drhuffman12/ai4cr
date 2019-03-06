require "./bp_net"

module Ai4cr
  module NeuralNetwork
    module Common
      module SimpleNetWithMemory
        include Common::BpNet

        property memory_qty
        property memory_node_set : Array(Array(Float64))

        def initialize(@structure : Array(Int32), disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil, memory_qty : Int32 = 1)
          @disable_bias = !!disable_bias
          @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? 0.25 : learning_rate.as(Float64)
          @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : 0.1
          # Below are set via #init_network, but must be initialized in the 'initialize' method to avoid being nilable:
          @activation_nodes = Array(Array(Float64)).new
          @weights = Array(Array(Array(Float64))).new
          @last_changes = Array(Array(Array(Float64))).new
          @deltas = Array(Array(Float64)).new

          @memory_qty = memory_qty >= 0 ? memory_qty : 1
          @memory_node_set = Array(Array(Float64)).new

          init_network
        end

        # Initialize (or reset) activation nodes and weights, with the
        # provided net structure and parameters.
        def init_network
          init_activation_nodes
          init_weights
          init_last_changes
          init_memory
          return self
        end
      end
    end
  end
end
