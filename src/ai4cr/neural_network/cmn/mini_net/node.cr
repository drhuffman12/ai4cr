require "json"
require "./common/calc_guess.cr"
require "./common/props_and_inits.cr"
require "./common/train_and_adjust.cr"
require "./../learning_style.cr"

# src/ai4cr/neural_network/cmn/mini_net/common/learning_style.cr
# src/ai4cr/neural_network/cmn/mini_net/node.cr

# irb -r "./src/ai4cr.cr"

# np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(2,3,LS_PRELU)
# nr1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(2,3,LS_RELU)
# ns1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(2,3,LS_SIGMOID)
# nt1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.new(2,3,LS_TANH)

# np1_json = np1.to_json
# nr1_json = nr1.to_json
# ns1_json = ns1.to_json
# nt1_json = nt1.to_json

# np1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.from_json(np1_json)
# nr1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.from_json(nr1_json)
# nd1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.from_json(ns1_json)
# nt1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Node.from_json(nt1_json)

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        class Node
          include JSON::Serializable

          # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-classes to be more manageable
          include Common::PropsAndInits
          include Common::CalcGuess
          include Common::TrainAndAdjust

          # property learning_style : Common::LearningStyle

          # @learning_style = Common::LearningStyle::Relu

          def propagation_function
            case @learning_style
            when LA_PRELU # LearningStyle::Prelu
              propagation_function_prelu
            when LA_RELU # LearningStyle::Rel
              propagation_function_relu
            when LA_SIGMOID # LearningStyle::Sigmoid
              propagation_function_sigmoid
            when LA_TANH # LearningStyle::Tanh
              propagation_function_tanh
            # else
            #   raise "Unsupported LearningStyle"
            end
          end

          def derivative_propagation_function
            case @learning_style
            when LA_PRELU # LearningStyle::Prelu
              derivative_propagation_function_prelu
            when LA_RELU # LearningStyle::Rel
              derivative_propagation_function_relu
            when LA_SIGMOID # LearningStyle::Sigmoid
              derivative_propagation_function_sigmoid
            when LA_TANH # LearningStyle::Tanh
              derivative_propagation_function_tanh
            # else
            #   raise "Unsupported LearningStyle"
            end
          end

          def guesses_best
            case @learning_style
            when LA_PRELU # LearningStyle::Prelu
              guesses_best_prelu
            when LA_RELU # LearningStyle::Rel
              guesses_best_relu
            when LA_SIGMOID # LearningStyle::Sigmoid
              guesses_best_sigmoid
            when LA_TANH # LearningStyle::Tanh
              guesses_best_tanh
            # else
            #   raise "Unsupported LearningStyle"
            end
          end

          # Prelu:
          @deriv_scale = 0.001

          def set_deriv_scale_prelu(scale)
            @deriv_scale = scale
          end

          def propagation_function_prelu
            ->(x : Float64) { x < 0 ? 0.0 : x }
          end

          def derivative_propagation_function_prelu
            ->(y : Float64) { y < 0 ? @deriv_scale : 1.0 }
          end

          def guesses_best_prelu
            guesses_ceiled
          end

          # Relu:
          def propagation_function_relu
            ->(x : Float64) { x < 0 ? 0.0 : [1.0, x].min }
          end

          def derivative_propagation_function_relu
            ->(y : Float64) { y < 0 ? 0.001 : 1.0 }
          end

          def guesses_best_relu
            guesses_ceiled
          end

          # Sigmoid:
          def propagation_function_sigmoid
            ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
          end

          def derivative_propagation_function_sigmoid
            ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
          end

          def guesses_best_sigmoid
            guesses_rounded
          end

          # Tanh:
          def propagation_function_tanh
            ->(x : Float64) { Math.tanh(x) }
          end

          def derivative_propagation_function_tanh
            ->(y : Float64) { 1.0 - (y**2) }
          end

          def guesses_best_tanh
            guesses_rounded
          end

        end
      end
    end
  end
end
