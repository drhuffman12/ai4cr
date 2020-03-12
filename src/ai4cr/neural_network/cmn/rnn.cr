require "json"
require "./learning_style.cr"
require "./mini_net_concerns/*"
require "./rnn_concerns/*"

# require "./rnn_concerns/net_config.cr"
# require "./rnn_concerns/mini_net_config.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      class Rnn
        include JSON::Serializable

        # NOTE: The first net should have a bias; the others should not.
        # TODO: Force bias only on 1st and none on others
        include RnnConcerns::PropsAndInits
        include RnnConcerns::CalcGuess
        include RnnConcerns::TrainAndAdjust
      end
    end
  end
end

# icr -r ./src/ai4cr.cr

# rnn = Ai4cr::NeuralNetwork::Cmn::Rnn.new
