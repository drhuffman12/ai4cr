require "ascii_bar_charter"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleTeamConcerns
        module ErrorDistanceHistory
          # Calculate the radius of the error as if each output cell is an value in a coordinate set
          # def calculate_error_distance_history
          #   return @error_distance_history = [-1.0] if @error_distance_history_max < 1
          #   if @error_distance_history.size < @error_distance_history_max - 1
          #     # Array not 'full' yet, so add latest value to end
          #     @error_distance_history << @error_distance
          #   else
          #     # Array 'full', so rotate end to front and then put new value at last index
          #     @error_distance_history.rotate!
          #     @error_distance_history[-1] = @error_distance
          #   end

          #   @error_distance_history_score = error_distance_history.map_with_index do |e, i|
          #     e / (2.0 ** (@error_distance_history_max - i))
          #   end.sum

          #   @error_distance_history
          # end

          def error_distance_history
            team_members.map do |rnn_simple|
              rnn_simple.error_distance_history
            end
          end

          def plot_error_distance_history(
            min = 0.0,
            max = 1.0,
            precision = 2.to_i8,
            in_bw = false,
            prefixed = false,
            reversed = false,
          )
            team_members.map do |rnn_simple|
              charter = AsciiBarCharter.new(min: min, max: max, precision: precision, in_bw: in_bw, inverted_colors: reversed)
              charter.plot(rnn_simple.error_distance_history, prefixed)
            end
          end
        end
      end
    end
  end
end
