require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module ConnectedNetSet
        class Chain
          include JSON::Serializable

          getter structure : Array(Int32)
          property net_set : Array(MiniNet::Common::AbstractNet)

          # NOTE: When passing in the array for net_set,
          # .. if you're including just one type of MiniNet, e.g.:
          #   net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: 256, width: 300, error_distance_history_max: 60)
          #   net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: 300, width: 3, error_distance_history_max: 60)
          #
          # ... and you try to pass in like below, you'll get a type error:
          #   cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new([net0, net1])
          #
          # ... So, you'll need to init the array like:
          #   arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet::Common::AbstractNet).new
          #   arr << net0
          #   arr << net1
          #
          # ... and then pass it in like:
          #   cns = Ai4cr::NeuralNetwork::Cmn::ConnectedNetSet::Chain.new(arr)
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
            @net_set.map_with_index do |net, index|
              net.height
            end << @net_set.last.width
          end

          def eval(inputs_given)
            @net_set.each_with_index do |net, index|
              index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@net_set[index - 1].outputs_guessed)
              net.step_calc_forward
            end

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

            @net_set.last.error_total
          end

          def guesses_best
            @net_set.last.guesses_best
          end

          def step_calculate_error_distance_history
            @net_set.last.step_calculate_error_distance_history
          end

          def error_distance_history
            @net_set.last.error_distance_history
          end
        end
      end
    end
  end
end
