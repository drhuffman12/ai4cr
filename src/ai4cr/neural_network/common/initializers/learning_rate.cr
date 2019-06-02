# Ai4cr::NeuralNetwork::Backpropagation::Common::Initializers::LearningRate

module Ai4cr
  module NeuralNetwork
    module Backpropagation
      module Common
        module Initializers
          module LearningRate
            DEFAULT_LEARNING_RATE = 0.25
    
            property learning_rate : Float64
  
            def init_learning_rate(_learning_rate)
              # must be positive
              _learning_rate.nil? || _learning_rate.as(Float64) <= 0.0 ? DEFAULT_LEARNING_RATE : _learning_rate.as(Float64)
            end  
          end
        end
      end
    end
  end
end
