require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        abstract class AbstractNet
          include JSON::Serializable

          abstract def propagation_function

          abstract def derivative_propagation_function

          abstract def guesses_best
          
          def guesses_rounded
            "guesses_rounded"
          end

          def guesses_ceiled
            "guesses_ceiled"
          end

          property foo : Int32
          property bar : Int32
          def initialize(@foo = 0, @bar = -1)
          end
        end
      end
    end
  end
end
