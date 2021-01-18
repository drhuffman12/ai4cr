require "./rnn_simple.cr"
require "./rnn_simple_team_concerns/props_and_inits.cr"
# require "./mini_net_concerns/error_distance_history.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleTeam
        # Team of Simple RNN's w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params
        include JSON::Serializable

        include RnnSimpleTeamConcerns::PropsAndInits
        # include MiniNetConcerns::ErrorDistanceHistory
      end
    end
  end
end
