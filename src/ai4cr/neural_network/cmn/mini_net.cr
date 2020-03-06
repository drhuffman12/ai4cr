require "json"
require "./common/calc_guess.cr"
require "./common/props_and_inits.cr"
require "./common/train_and_adjust.cr"
require "./learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module MiniNet
        class MiniNet
          include JSON::Serializable

          # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-classes to be more manageable
          include Common::PropsAndInits
          include Common::CalcGuess
          include Common::TrainAndAdjust
        end
      # end
    end
  end
end
