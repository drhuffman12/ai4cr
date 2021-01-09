require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module SplitTrainingData
          def split_for_training(training_data) : TrainingData
            # def split_for_training(training_data : Array(Array(Float64)))
            # TODO: slice up for easier more granular testing

            training_data_size = training_data.size
            io_pairs_tc_size = io_offset + time_col_qty
            io_pairs_qty = training_data_size - io_pairs_tc_size + 1
            (0..io_pairs_qty - 1).to_a.map do |i|
              from = i
              to = i + time_col_qty - 1
              ins = training_data[from..to]

              from = io_offset + i
              to = io_offset + i + time_col_qty - 1
              outs = training_data[from..to]

              {
                ins:  ins,
                outs: outs,
              }
            end
          end

          def float_to_state_values(values, to_min_i = 0, to_max_i = 20, from_min = -1.0, from_max = 1.0)
            # This is for when you have data in float values [Array(Float64), or Array(Int32)],
            # but you want to convert to to single-state representation
            # [ideally Array(Array(Int32)), but, in our case, Array(Array(Float64))]

            from_denom = (from_max - from_min)
            to_denom = (to_max_i - to_min_i)
            values.map do |v|
              from_percent = 1.0 * (v - from_min) / from_denom
              to_v = (0.0 + to_min_i + from_percent * to_denom).round.to_i
              to_v = case
                     when to_v < to_min_i
                       to_min_i
                     when to_v > to_max_i
                       to_max_i
                     else
                       to_v
                     end

              (to_min_i..to_max_i).to_a.map do |i|
                i == to_v ? 1.0 : 0.0
              end
            end
          end
        end
      end
    end
  end
end
