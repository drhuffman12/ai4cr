require "./rnn_simple.cr"
require "./rnn_simple_team_concerns/props_and_inits.cr"
require "./rnn_simple_team_concerns/calc_guess.cr"
require "./rnn_simple_team_concerns/error_distance_history.cr"
# require "./rnn_simple_team_concerns/train_and_adjust.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleTeam
        # Team of Simple RNN's w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params
        include JSON::Serializable

        include RnnSimpleTeamConcerns::PropsAndInits
        include RnnSimpleTeamConcerns::CalcGuess
        include RnnSimpleTeamConcerns::ErrorDistanceHistory
        # include RnnSimpleTeamConcerns::TrainAndAdjust
      end
    end
  end
end
