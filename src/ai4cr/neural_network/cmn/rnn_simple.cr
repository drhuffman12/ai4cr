require "json"
require "./learning_style.cr"
require "./rnn_concerns/calc_guess.cr"
require "./rnn_concerns/props_and_inits.cr"
require "./rnn_concerns/train_and_adjust.cr"
require "./rnn_concerns/roll_ups.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class RnnSimple
        # EMPTY_1D_ARRAY_FLOAT64 = Array(Float64).new

        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

        include JSON::Serializable

        include RnnConcerns::PropsAndInits
        include RnnConcerns::CalcGuess
        include RnnConcerns::TrainAndAdjust
        include RnnConcerns::RollUps
      end
    end
  end
end
