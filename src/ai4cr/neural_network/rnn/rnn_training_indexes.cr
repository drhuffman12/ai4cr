module Ai4cr
  module NeuralNetwork
    module Rnn
      alias RnnTrainingIndexes = NamedTuple(
        training_in_indexes: Array(NamedTuple(i_from: Int32, i_to: Int32)),
        training_out_indexes: Array(NamedTuple(i_from: Int32, i_to: Int32)),
        next_eval_in_indexes: NamedTuple(i_from: Int32, i_to: Int32))
    end
  end
end
