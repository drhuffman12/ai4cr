require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        module Common
          module PropsAndInits
            getter width : Int32, height : Int32
            getter height_considering_bias : Int32
            getter width_indexes : Array(Int32), height_indexes : Array(Int32)
            property inputs_given : Array(Float64), outputs_guessed : Array(Float64)
            property weights : Array(Array(Float64))
            property last_changes : Array(Array(Float64)) # aka previous weights
            property error_total : Float64

            property outputs_expected : Array(Float64)

            property input_deltas : Array(Float64), output_deltas : Array(Float64)

            property disable_bias : Bool
            property learning_rate : Float64
            property momentum : Float64

            getter error_distance : Float64
            getter error_distance_history_max : Int32
            getter error_distance_history : Array(Float64)

            def initialize(
              @height, @width,
              disable_bias : Bool? = nil, learning_rate : Float64? = nil, momentum : Float64? = nil,
              error_distance_history_max : Int32 = 10
            )
              @disable_bias = !!disable_bias
              @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
              @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

              # init_network:
              @height_considering_bias = @height + (@disable_bias ? 0 : 1)
              @height_indexes = Array.new(@height_considering_bias) { |i| i }

              @inputs_given = Array.new(@height_considering_bias, 0.0)
              @inputs_given[-1] = 1 unless @disable_bias
              @input_deltas = Array.new(@height_considering_bias, 0.0)

              @width_indexes = Array.new(width) { |i| i }

              @outputs_guessed = Array.new(width, 0.0)
              @outputs_expected = Array.new(width, 0.0)
              @output_deltas = Array.new(width, 0.0)

              @weights = @height_indexes.map { @width_indexes.map { rand*2 - 1 } }

              @last_changes = Array.new(@height_considering_bias, Array.new(width, 0.0))

              @error_total = 0.0
              @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
              @error_distance = 1.0
              @error_distance_history = Array.new(0, 0.0)
            end

            def init_network(error_distance_history_max : Int32 = 10)
              # init_network:
              @height_considering_bias = @height + (@disable_bias ? 0 : 1)
              @height_indexes = Array.new(@height_considering_bias) { |i| i }

              @inputs_given = Array.new(@height_considering_bias, 0.0)
              @inputs_given[-1] = 1 unless @disable_bias
              @input_deltas = Array.new(@height_considering_bias, 0.0)

              @width_indexes = Array.new(width) { |i| i }

              @outputs_guessed = Array.new(width, 0.0)
              @outputs_expected = Array.new(width, 0.0)
              @output_deltas = Array.new(width, 0.0)

              @weights = @height_indexes.map { @width_indexes.map { rand*2 - 1 } }

              @last_changes = Array.new(@height_considering_bias, Array.new(width, 0.0))

              @error_total = 0.0
              @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
              @error_distance = 0.0
              @error_distance_history = Array.new(0, 0.0)
            end

            def structure
              [height, width]
            end
          end
        end
      end
    end
  end
end
