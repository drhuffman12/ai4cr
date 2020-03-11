require "json"
require "./learning_style.cr"
require "./mini_net_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class MiniNet
        include JSON::Serializable

        # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-classes to be more manageable
        include MiniNetConcerns::PropsAndInits
        include MiniNetConcerns::CalcGuess
        include MiniNetConcerns::TrainAndAdjust
      end
    end
  end
end
