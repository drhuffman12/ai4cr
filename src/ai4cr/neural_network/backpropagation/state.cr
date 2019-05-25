require "json"

module Ai4cr
  module NeuralNetwork
    module Backpropagation
      struct State
        include JSON::Serializable
        include Ai4cr::NeuralNetwork::Backpropagation::Math

        property config : Config

        property weights : Array(Array(Array(Float64)))
        property last_changes : Array(Array(Array(Float64)))
        property activation_nodes : Array(Array(Float64))
        property deltas : Array(Array(Float64))
        property input_deltas : Array(Float64)
        property calculated_error_total : Float64

        # @activation_nodes : Array(Array(Float64))
        # @weights : Array(Array(Array(Float64)))
        # @last_changes : Array(Array(Array(Float64)))
        # @deltas : Array(Array(Float64))
        # @input_deltas : Array(Float64)

        # def initialize(@config)
        def initialize(structure : Array(Int32), disable_bias : Bool? = true, learning_rate : Float64? = nil, momentum : Float64? = nil)
          @config = Config.new(structure, disable_bias, learning_rate, momentum)

          @activation_nodes = init_activation_nodes
          @weights = init_weights
          @last_changes = init_last_changes
          @deltas = init_deltas
          @input_deltas = init_input_deltas
          @calculated_error_total = 0.0
        end

        # TODO: Remove
        # Initialize (or reset) activation nodes and weights, with the
        # provided net structure and parameters.
        def init_network
          init_activation_nodes
          init_weights
          init_last_changes
          init_deltas
          return self
        end

        # Initialize neurons structure.
        private def init_activation_nodes
          act_nodes = (0...config.structure.size).map do |n|
            (0...config.structure[n]).map { 1.0 }
          end
          if !config.disable_bias
            act_nodes[0...-1].each { |layer| layer << 1.0 }
          end
          act_nodes
        end

        # Initialize the weight arrays using function specified with the
        # initial_weight_function parameter
        private def init_weights
          (0...config.structure.size - 1).map do |i|
            nodes_origin_size = @activation_nodes[i].size
            nodes_target_size = config.structure[i + 1]
            (0...nodes_origin_size).map do |j|
              (0...nodes_target_size).map do |k|
                initial_weight_function.call(i, j, k)
              end
            end
          end
        end

        # Momentum usage need to know how much a weight changed in the
        # previous training. This method initialize the @last_changes
        # structure with 0 values.
        private def init_last_changes
          (0...@weights.size).map do |w|
            (0...@weights[w].size).map do |i|
              (0...@weights[w][i].size).map { 0.0 }
            end
          end
        end

        private def init_deltas
          config.structure.map{|layer_size| layer_size.times.map{0.0}.to_a}.to_a
        end
        
        private def init_input_deltas
          config.structure.first.times.map{0.0}.to_a
        end

      end
    end
  end
end