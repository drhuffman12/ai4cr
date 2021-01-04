require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module SplitTrainingData
          def split(training_data, eval_qty = 1)
            # TODO: slice up for easier more granular testing

            training_data_size = training_data.size
            io_pairs_tc_size = io_offset + time_col_qty
            io_pairs_qty = training_data_size - io_pairs_tc_size + 1
            io_pairs_indexes = Array.new(io_pairs_qty) { |i| i }
            io_sets_pairs = io_pairs_indexes.map do |i|
              from = i
              to = i + time_col_qty - 1
              ins = training_data[from..to] # .clone

              from = io_offset + i
              to = io_offset + i + time_col_qty - 1
              outs = training_data[from..to] # .clone

              {
                ins:  ins,
                outs: outs,
              }
            end

            # io_sets_size = io_sets_pairs.size

            eval_to = io_pairs_qty - 1
            eval_from = eval_to - eval_qty + 1
            train_to = eval_from - 1
            train_from = 0

            io_sets_train = io_sets_pairs[train_from..train_to]
            io_sets_eval = io_sets_pairs[eval_from..eval_to]

            {
              training_data_size: training_data_size,
              io_pairs_tc_size:   io_pairs_tc_size,
              io_pairs_qty:       io_pairs_qty,
              io_pairs_indexes:   io_pairs_indexes,

              training_qty: io_sets_train.size,
              eval_qty:     io_sets_eval.size,

              # io_sets_pairs: io_sets_pairs,
              io_sets_train: io_sets_train,
              io_sets_eval:  io_sets_eval,
            }
          end

          # def split_as_all_training(training_data)
          # end

          # def split_as_all_eval(training_data)
          # end
        end
      end
    end
  end
end
