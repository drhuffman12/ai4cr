
require "./../../breed_parent.cr"
require "./mini_net_concerns/breed_parent.cr"

require "./mini_net_concerns/calc_guess.cr"
require "./mini_net_concerns/props_and_inits.cr"
require "./mini_net_concerns/train_and_adjust.cr"
# require "./mini_net_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class MiniNet # < Ai4cr::NeuralNetwork::Cmn::MiniNetConcerns::BreedParent
        include JSON::Serializable

        # include Ai4cr::BreedParent(self.class)
        include Ai4cr::BreedParent
        include MiniNetConcerns::BreedParent

        # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-/related-classes to be more manageable
        include MiniNetConcerns::PropsAndInits
        include MiniNetConcerns::CalcGuess
        include MiniNetConcerns::TrainAndAdjust

      end
    end
  end
end
