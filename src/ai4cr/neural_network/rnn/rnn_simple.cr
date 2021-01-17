require "./rnn_training_indexes.cr"
require "./rnn_concerns/calc_guess.cr"
require "./rnn_concerns/props_and_inits.cr"
require "./rnn_concerns/train_and_adjust.cr"
require "./rnn_concerns/roll_ups.cr"
require "./rnn_concerns/data_utils.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimple
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params
        include JSON::Serializable

        include RnnConcerns::PropsAndInits
        include RnnConcerns::CalcGuess
        include RnnConcerns::TrainAndAdjust
        include RnnConcerns::RollUps
        include RnnConcerns::DataUtils
      end
    end
  end
end
