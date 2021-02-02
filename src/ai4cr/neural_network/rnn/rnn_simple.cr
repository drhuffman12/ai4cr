require "./rnn_simple_concerns/calc_guess.cr"
require "./rnn_simple_concerns/props_and_inits.cr"
require "./rnn_simple_concerns/train_and_adjust.cr"
require "./rnn_simple_concerns/roll_ups.cr"
require "./rnn_simple_concerns/data_utils.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimple
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params

        include JSON::Serializable

        include RnnSimpleConcerns::PropsAndInits
        include RnnSimpleConcerns::CalcGuess
        include RnnSimpleConcerns::TrainAndAdjust
        include RnnSimpleConcerns::RollUps
        include RnnSimpleConcerns::DataUtils
        include Ai4cr::Breed::Client
      end
    end
  end
end
