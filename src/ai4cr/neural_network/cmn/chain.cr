require "json"
require "./learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module ConnectedNetSet
      class Chain
        # NOTE: The first net should have a bias; the others should not.
        # TODO: Force bias only on 1st and none on others

        include JSON::Serializable

        getter structure : Array(Int32)
        property net_set : Array(MiniNet)
        getter net_set_size : Int32
        getter net_set_indexes_reversed : Array(Int32)
        getter weight_height_mismatches : Array(Hash(Symbol, Int32))

        # NOTE: When passing in the array for net_set,
        # .. if you're including just one type of MiniNet, e.g.:
        #   net0 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: 256, width: 300, error_distance_history_max: 60)
        #   net1 = Ai4cr::NeuralNetwork::Cmn::MiniNet::Sigmoid.new(height: 300, width: 3, error_distance_history_max: 60)
        #
        # ... and you try to pass in like below, you'll get a type error:
        #   cns = Ai4cr::NeuralNetwork::Cmn::Chain.new([net0, net1])
        #
        # ... So, you'll need to init the array like:
        #   arr = Array(Ai4cr::NeuralNetwork::Cmn::MiniNet).new
        #   arr << net0
        #   arr << net1
        #
        # ... and then pass it in like:
        #   cns = Ai4cr::NeuralNetwork::Cmn::Chain.new(arr)
        def initialize(@net_set)
          @structure = calc_structure
          @net_set_size = @net_set.size
          @net_set_indexes_reversed = Array.new(@net_set_size) { |i| @net_set_size - i - 1}

          @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
        end

        def validate
          index_max = @net_set_size - 1

          @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
          @weight_height_mismatches = @net_set.map_with_index do |net_from, index|
            if index >= index_max
              nil # There is no 'next' net after the last net, so we don't need to compare any sizes
            else
              net_to = @net_set[index + 1]
              if net_from.width != net_to.height_considering_bias
                {
                  :from_index                 => index,
                  :to_index                   => index + 1,
                  :from_width                 => net_from.width,
                  :to_height_considering_bias => @net_set[index + 1].height_considering_bias,
                  :from_disable_bias          => (net_from.disable_bias ? 1 : 0),
                  :to_disable_bias            => (@net_set[index + 1].disable_bias ? 1 : 0),
                }
              end
            end
          end.compact

          @weight_height_mismatches.any? ? false : true
        end

        def errors
          @weight_height_mismatches
        end

        def validate!
          validate ? true : raise "Invalid net set (width vs height mismatch), errors: #{errors}"
        end

        def calc_structure
          @net_set.map do |net|
            net.height
          end << @net_set.last.width
        end

        def eval(inputs_given)
          @net_set.each_with_index do |net, index|
            if index == 0
              net.step_load_inputs(inputs_given)
            else
              net.step_load_inputs(@net_set[index - 1].outputs_guessed)
            end

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

          index_max = @net_set_size - 1
          @net_set_indexes_reversed.each do |index|
            net = @net_set[index]

            index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@net_set[index + 1].input_deltas)

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
