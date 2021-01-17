module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnConcerns
        module DataUtils
          def indexes_for_training_and_eval(training_data) # training_data # : TrainingData
            # TODO: slice up for easier more granular testing
            # This leads into io data via `training_data[from..to]`

            training_data_size = training_data.size
            io_pairs_tc_size = io_offset + time_col_qty
            io_pairs_qty = training_data_size - io_pairs_tc_size + 1

            training_in_indexes = (0..io_pairs_qty - 1).to_a.map do |i|
              {i_from: i, i_to: i + time_col_qty - 1}
            end

            training_out_indexes = (0..io_pairs_qty - 1).to_a.map do |i|
              {i_from: io_offset + i, i_to: io_offset + i + time_col_qty - 1}
            end

            i = io_pairs_qty
            next_eval_in_indexes = {i_from: i, i_to: i + time_col_qty - 1}

            {
              training_in_indexes:  training_in_indexes,
              training_out_indexes: training_out_indexes,
              next_eval_in_indexes: next_eval_in_indexes,
            }
          end

          # def append_next_sensor_data(next_sensor_data : Array(Float64))
          #   @training_data << next_sensor_data
          # end

          # def set_training_and_eval_sets(training_data)
          #   split_for_training_data(training_data)

          # end

          def float_to_state_values(values, to_min_i = 0, to_max_i = 20, from_min = -1.0, from_max = 1.0)
            # TODO: slice up for easier more granular testing

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
