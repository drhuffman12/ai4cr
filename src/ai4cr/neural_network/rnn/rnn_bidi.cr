require "./rnn_simple.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnBiDi < RnnSimple
        # TODO: Implement Bi-directional RNN (i.e.: RnnSimple pulls from inputs and previous time column.)
        # This class must also pull from next time column and mix them all together in subsequent hidden layers.
      end
    end
  end
end
