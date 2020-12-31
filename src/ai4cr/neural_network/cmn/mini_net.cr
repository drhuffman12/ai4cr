require "json"
require "./mini_net_concerns/calc_guess.cr"
require "./mini_net_concerns/props_and_inits.cr"
require "./mini_net_concerns/train_and_adjust.cr"
require "./learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module MiniNet
      class MiniNet
        include JSON::Serializable

        # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-classes to be more manageable
        include MiniNetConcerns::PropsAndInits
        include MiniNetConcerns::CalcGuess
        include MiniNetConcerns::TrainAndAdjust
      end
      # end
    end
  end
end
