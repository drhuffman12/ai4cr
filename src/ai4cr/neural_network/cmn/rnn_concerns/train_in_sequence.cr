require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module TrainInSequence
          def train_and_guess_in_sequence(io_pairs)
            # io_pairs = split_for_training(training_data)

            train_in_sequence(io_pairs)

            next_inputs = shifted_inputs(io_pairs.last)

            eval(next_inputs)
          end

          def train_in_sequence(io_pairs)
            io_pairs.map do |io_pair|
              train(io_pair[:ins], io_pair[:outs])
            end
          end

          def shifted_inputs(io_pair)
            io_pair[:ins][1..-1] << io_pair[:outs][0]
          end
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
