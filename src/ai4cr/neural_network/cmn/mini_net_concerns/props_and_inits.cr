require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        module PropsAndInits
          getter width : Int32, height : Int32
          getter height_considering_bias : Int32
          getter width_indexes : Array(Int32), height_indexes : Array(Int32)
          property inputs_given : Array(Float64), outputs_guessed : Array(Float64)
          property weights : Array(Array(Float64))
          property last_changes : Array(Array(Float64)) # aka previous weights
          property error_total : Float64

          property outputs_expected : Array(Float64)

          property input_deltas : Array(Float64)
          property output_deltas : Array(Float64)

          property bias_disabled : Bool
          property bias_scale : Float64
          property learning_rate : Float64
          property momentum : Float64

          getter error_distance : Float64
          getter error_distance_history_max : Int32
          getter error_distance_history : Array(Float64)

          property learning_style : LearningStyle
          property deriv_scale : Float64

          def initialize(mini_net_config : MiniNetConcerns::Config)
            @height = mini_net_config.height
            @width = mini_net_config.width
            @learning_style = mini_net_config.learning_style
            @deriv_scale = mini_net_config.deriv_scale
            @bias_disabled = !!mini_net_config.bias_disabled
            @bias_scale = (mini_net_config.bias_scale < 0.0) ? [1.0, mini_net_config.bias_scale].min : 0.0
            @learning_rate = mini_net_config.learning_rate.nil? || mini_net_config.learning_rate.as(Float64) <= 0.0 ? rand : [1.0, mini_net_config.learning_rate.as(Float64)].min
            @momentum = mini_net_config.momentum && mini_net_config.momentum.as(Float64) > 0.0 ? [1.0, mini_net_config.momentum.as(Float64)].min : rand

            @error_distance_history_max = (mini_net_config.error_distance_history_max < 0 ? 0 : mini_net_config.error_distance_history_max)

            # TODO: consolidate 'init_network' code
            # init_network:
            @height_considering_bias = @height + (@bias_disabled ? 0 : 1)
            @height_indexes = Array.new(@height_considering_bias) { |i| i }

            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = 1.0 unless @bias_disabled
            # @inputs_given[-1] = 0.1 unless @bias_disabled
            @input_deltas = Array.new(@height_considering_bias, 0.0)

            @width_indexes = Array.new(width) { |i| i }

            @outputs_guessed = Array.new(width, 0.0)
            @outputs_expected = Array.new(width, 0.0)
            @output_deltas = Array.new(width, 0.0)

            # TODO: set weights based on learning_type
            # See: https://medium.com/datadriveninvestor/deep-learning-best-practices-activation-functions-weight-initialization-methods-part-1-c235ff976ed
            # @weights = @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(2.0/(height_considering_bias + width))) } }
            @weights = case @learning_style
                       when LS_TANH, LS_SIGMOID
                         # Xavier initialization mostly used with tanh and logistic activation function
                         @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(2.0/(height_considering_bias + width))) } }
                       when LS_RELU, LS_PRELU
                         # He-initialization mostly used with ReLU or it’s variants — Leaky ReLU.
                         @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(height_considering_bias/2.0)) / 100.0 } }
                       else
                        #  raise "Unsupported Learning Style: #{@learning_style}"
                         @height_indexes.map { @width_indexes.map { rand*2 - 1 } }
                       end

            @last_changes = Array.new(@height_considering_bias, Array.new(width, 0.0))

            @error_total = 0.0
            @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
            @error_distance = 1.0
            @error_distance_history = Array.new(0, 0.0)
          end

          def initialize(
            @height = 2, @width = 1,
            @learning_style : LearningStyle = LS_RELU, #  LearningStyle::Relu,

            # for Prelu
            # TODO: set deriv_scale based on ?
            # @deriv_scale = 0.1,
            # @deriv_scale = 0.01,
            # @deriv_scale = 0.001,
            @deriv_scale = rand / 100.0,

            bias_disabled : Bool? = nil,
            @bias_scale : Float64 = rand,

            learning_rate : Float64? = nil,
            momentum : Float64? = nil,
            error_distance_history_max : Int32 = 10
          )
            # @learning_style = Common::LearningStyle::Relu

            @bias_disabled = !!bias_disabled
            # keep bias_scale between -1 and 1
            @bias_scale = (bias_scale < -1.0) ? -1.0 : [1.0, bias_scale].min
            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : [1.0, learning_rate.as(Float64)].min
            @momentum = momentum && momentum.as(Float64) > 0.0 ? [1.0, momentum.as(Float64)].min : rand

            # init_network:
            @height_considering_bias = @height + (@bias_disabled ? 0 : 1)
            @height_indexes = Array.new(@height_considering_bias) { |i| i }

            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = bias_scale unless @bias_disabled
            @input_deltas = Array.new(@height_considering_bias, 0.0)

            @width_indexes = Array.new(width) { |i| i }

            @outputs_guessed = Array.new(width, 0.0)
            @outputs_expected = Array.new(width, 0.0)
            @output_deltas = Array.new(width, 0.0)

            # TODO: set weights based on learning_type
            # @weights = @height_indexes.map { @width_indexes.map { rand*2 - 1 } }
            # @weights = @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(2.0/(height_considering_bias + width))) } }
            @weights = @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(height_considering_bias/2.0)) } }

            @last_changes = Array.new(@height_considering_bias, Array.new(width, 0.0))

            @error_total = 0.0
            @error_distance_history_max = (error_distance_history_max < 0 ? 0 : error_distance_history_max)
            @error_distance = 1.0
            @error_distance_history = Array.new(0, 0.0)
          end

          def init_network(error_distance_history_max : Int32 = 10)
            # init_network:
            @height_considering_bias = @height + (@bias_disabled ? 0 : 1)
            @height_indexes = Array.new(@height_considering_bias) { |i| i }

            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = 1.0 unless @bias_disabled
            # @inputs_given[-1] = 0.1 unless @bias_disabled
            @input_deltas = Array.new(@height_considering_bias, 0.0)

            @width_indexes = Array.new(width) { |i| i }

            @outputs_guessed = Array.new(width, 0.0)
            @outputs_expected = Array.new(width, 0.0)
            @output_deltas = Array.new(width, 0.0)

            # Weight initialization (https://medium.com/datadriveninvestor/deep-learning-best-practices-activation-functions-weight-initialization-methods-part-1-c235ff976ed)
            # * Xavier initialization mostly used with tanh and logistic activation function
            # * He-initialization mostly used with ReLU or it’s variants — Leaky ReLU.

            @weights = @height_indexes.map { @width_indexes.map { rand*2 - 1 } }
            # @weights = @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(2.0/(height_considering_bias + width))) } }
            # @weights = @height_indexes.map { @width_indexes.map { (rand*2 - 1)*(Math.sqrt(height_considering_bias/2.0)) } }

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
