require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainInSequence
          # def train_and_guess_in_sequence(io_pairs, training_rounds = 1, guess_ahead_qty = 3)
          #   # io_pairs = split_for_training(training_data)
          #   # @io_pairs = io_pairs

          #   training_rounds.times.each { train_in_sequence(io_pairs) }

          #   io_pairs_last = io_pairs.last

          #   guess_ahead_qty.times.to_a.map do
          #     next_inputs = shifted_inputs(io_pairs_last)

          #     last_guess_set = eval(next_inputs)

          #     next_outputs = shifted_outputs(io_pairs_last)

          #     io_pairs_last = {
          #       ins:  next_inputs,
          #       outs: next_outputs,
          #     }
          #     # io_pairs_last[:ins] = next_inputs
          #     # io_pairs_last[:outs] = next_outputs

          #     last_guess_set
          #   end
          # end

          # def train_in_sequence(io_pairs)
          #   io_pairs.map do |io_pair|
          #     train(io_pair[:ins], io_pair[:outs])
          #   end
          # end

          # def shifted_inputs(io_pair)
          #   io_pair[:ins][1..-1] << io_pair[:outs][0]
          # end

          # def shifted_outputs(io_pair)
          #   latest_guess = mini_net_set[synaptic_layer_index_last][time_col_index_last].outputs_guessed
          #   io_pair[:outs][1..-1] << latest_guess
          # end
        end
      end
    end
  end
end

# def train_in_sequence(split_training_data)
#   # train which returns error values
#   training_errors = split_training_data[:io_sets_train].map do |io_sets|
#     train(io_sets[:ins], io_sets[:outs])
#   end

#   # eval and compare which returns error values
#   # guessing_errors = split_training_data[:io_sets_eval].map do |io_sets|
#   #   # guessed = eval(io_sets[:ins])

#   #   # expected = puts io_sets[:outs]

#   #   # # block.call(guessed, expected)
#   #   # yield(guessed, expected)
#   #   eval_and_compare(io_sets[:ins], io_sets[:outs])
#   # end

#   # {
#   #   training_errors: training_errors,
#   #   guessing_errors: guessing_errors
#   # }
# end
