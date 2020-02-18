require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module ConnectedNetSet
        class Chain
          include JSON::Serializable

          getter structure : Array(Int32)
          property net_set : Array(MiniNet::Common::AbstractNet)

          def initialize(@net_set)
            @structure = calc_structure
          end

          def validate
            index_max = @net_set.size - 1

            width_height_mismatch = @net_set.map_with_index do |net_from, index|
              if index >= index_max
                false
              else
                net_to = @net_set[index + 1]
                net_from.width != net_to.height
              end
            end.any?

            width_height_mismatch ? false : true
          end

          def validate!
            validate ? true : raise "Invalid net set (width vs height mismatch)"
          end

          def calc_structure
            # [height, width]
            @net_set.map_with_index do |net, index|
              net.height
            end << @net_set.last.width
          end

          def eval(inputs_given) # aka eval
            @net_set.each_with_index do |net, index|
              index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@net_set[index - 1].outputs_guessed)
              net.step_calc_forward
            end
            # ...

            @net_set.last.outputs_guessed
          end

          # TODO: utilize until_min_avg_error
          def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
            @net_set.each_with_index do |net, index|
              index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@net_set[index - 1].outputs_guessed)
              net.step_calc_forward
            end

            index_max = @net_set.size - 1
            (0..index_max).to_a.reverse.each do |index|
              net = @net_set[index]

              index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@net_set[index + 1].input_deltas[0..@net_set[index + 1].height - 1])

              net.step_calculate_error
              net.step_backpropagate
            end

            @net_set.last.error_total # @error
          end

          def guesses_best
            @net_set.last.guesses_best
          end
        end
      end
    end
  end
end
