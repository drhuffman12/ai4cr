require "json"
require "./calc_guess.cr"
require "./props_and_inits.cr"
require "./train_and_adjust.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        module Common
          abstract class AbstractNet
            include JSON::Serializable

            # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-classes to be more manageable
            include PropsAndInits
            include CalcGuess
            include TrainAndAdjust
          end
        end
      end
    end
  end
end
