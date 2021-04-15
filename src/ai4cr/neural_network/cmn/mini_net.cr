# require "./../../breed_parent.cr"
# require "./mini_net_concerns/breed_parent.cr"

require "./mini_net_concerns/calc_guess.cr"
require "./mini_net_concerns/props_and_inits.cr"
require "./mini_net_concerns/train_and_adjust.cr"

# require "./mini_net_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class MiniNet
        include JSON::Serializable

        # MiniNet code (based on original ai4r Backpropagation) is split up into modules and abstract-/sub-/related-classes to be more manageable

        include Ai4cr::Breed::Client
        include MiniNetConcerns::PropsAndInits
        include MiniNetConcerns::CalcGuess
        include MiniNetConcerns::TrainAndAdjust

        def self.config_rand(
          name : String = Time.utc.to_s,
          height : Int32 = 2,
          width : Int32 = 2,
          learning_styles : LearningStyle = LEARNING_STYLES_DEFAULT,
          bias_disabled = false,
          history_size = 10
        )
          {
            height:          height,
            width:           width,
            learning_styles: learning_styles,

            deriv_scale: Ai4cr::Utils::Rand.rand_excluding(scale: 0.5),

            bias_disabled: bias_disabled,
            bias_default:  Ai4cr::Utils::Rand.rand_excluding,

            learning_rate: Ai4cr::Utils::Rand.rand_excluding,
            momentum:      Ai4cr::Utils::Rand.rand_excluding,
            history_size:  history_size,

            name: name,
          }
        end

        def clone
          a_clone = MiniNet.new(
            height: self.height, width: self.width,
            learning_styles: self.learning_styles,

            deriv_scale: self.deriv_scale,

            bias_disabled: self.bias_disabled, bias_default: self.bias_default,

            learning_rate: self.learning_rate, momentum: self.momentum,
            history_size: self.history_size,

            name: self.name
          )

          # calc_guess
          a_clone.weights = self.weights.clone
          a_clone.inputs_given = self.inputs_given.clone
          a_clone.outputs_guessed = self.outputs_guessed.clone

          # train_and_adjust
          a_clone.outputs_expected = self.outputs_expected.clone
          a_clone.output_deltas = self.output_deltas.clone
          a_clone.last_changes = self.last_changes.clone
          a_clone.output_errors = self.output_errors.clone
          a_clone.input_deltas = self.input_deltas.clone

          a_clone
        end
      end
    end
  end
end
