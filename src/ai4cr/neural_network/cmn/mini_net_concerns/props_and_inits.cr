module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        module PropsAndInits
          def initialize(
            @height, @width,
            @learning_style : LearningStyle = LS_RELU,

            @deriv_scale = rand / 2.0,

            disable_bias : Bool? = nil, @bias_default = 1.0,

            learning_rate : Float64? = nil, momentum : Float64? = nil,
            history_size : Int32 = 10
          )
            # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
            @disable_bias = disable_bias.nil? ? false : !!disable_bias

            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
            @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

            # init_network:
            init_network(history_size)

            @error_stats = Ai4cr::ErrorStats.new(history_size)
          end

          def structure
            [height, width]
          end

          def init_network(history_size : Int32 = 10)
            init_net_re_structure
            init_net_re_guess
            init_net_re_train(history_size)
          end
        end
      end
    end
  end
end
