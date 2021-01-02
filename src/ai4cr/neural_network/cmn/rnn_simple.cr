require "json"
require "./learning_style.cr"
require "./rnn_concerns/calc_guess.cr"
require "./rnn_concerns/props_and_inits.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class RnnSimple
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

        include JSON::Serializable

        include RnnConcerns::PropsAndInits
        include RnnConcerns::CalcGuess
        # include RnnConcerns::TrainAndAdjust
      end
    end
  end
end
