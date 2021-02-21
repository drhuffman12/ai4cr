# require "./../../breed_parent.cr"
# require "./mini_net_concerns/breed_parent.cr"

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
        # include Ai4cr::BreedParent
        # include MiniNetConcerns::BreedParent

        include Ai4cr::Breed::Client
        # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-/related-classes to be more manageable
        include MiniNetConcerns::PropsAndInits
        include MiniNetConcerns::CalcGuess
        include MiniNetConcerns::TrainAndAdjust

        def self.config_rand(
          name : String = Time.utc.to_s,
          height : Int32 = 2,
          width : Int32 = 2,
          learning_style : LearningStyle = LEARNING_STYLE_DEFAULT,
          bias_disabled = false,
          history_size = 10
        )
          {
            height:         height,
            width:          width,
            learning_style: learning_style,

            deriv_scale: Ai4cr::Data::Utils.rand_excluding(scale: 0.5),

            bias_disabled: bias_disabled,
            bias_default:  Ai4cr::Data::Utils.rand_excluding,

            learning_rate: Ai4cr::Data::Utils.rand_excluding,
            momentum:      Ai4cr::Data::Utils.rand_excluding,
            history_size:  history_size,

            name: name,
          }
        end
      end
    end
  end
end
