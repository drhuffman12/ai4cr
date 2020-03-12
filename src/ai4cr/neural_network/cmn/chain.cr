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
        property mini_net_set : Array(MiniNet)
        # getter weight_height_mismatches : type_of({from_index: 1, to_index: 2, from_width: 3, to_height_considering_bias: 4, from_bias: false, to_bias: true})
        # getter weight_height_mismatches : NamedTuple(from_index: Int32, to_index: Int32, from_width: Int32, to_height_considering_bias: Int32, from_bias: Bool, to_bias: Bool)
        # getter weight_height_mismatches : Array(Hash(Symbol,UInt32))
        getter weight_height_mismatches : Array(Hash(Symbol, Int32))

        # UINT_FALSE = 0_u32
        # UINT_TRUE = 1_u32

        # NOTE: When passing in the array for mini_net_set,
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
        def initialize(@mini_net_set)
          @structure = calc_structure
          # @weight_height_mismatches = Array(Hash(Symbol,UInt32)).new
          @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
        end

        def validate
          index_max = @mini_net_set.size - 1

          # @weight_height_mismatches = Array(Hash(Symbol,UInt32)).new
          @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
          @weight_height_mismatches = @mini_net_set.map_with_index do |net_from, index|
            if index >= index_max
              nil # There is no 'next' net after the last net, so we don't need to compare any sizes
            else
              net_to = @mini_net_set[index + 1]
              if net_from.width != net_to.height_considering_bias
                {
                  :from_index                 => index,
                  :to_index                   => index + 1,
                  :from_width                 => net_from.width,
                  :to_height_considering_bias => @mini_net_set[index + 1].height_considering_bias,
                  :from_disable_bias          => (net_from.bias_disabled ? 1 : 0),
                  :to_disable_bias            => (@mini_net_set[index + 1].bias_disabled ? 1 : 0),
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
          @mini_net_set.map do |net|
            net.height
          end << @mini_net_set.last.width
        end

        def eval(inputs_given)
          @mini_net_set.each_with_index do |net, index|
            # index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)

            # load inputs
            if index == 0
              net.step_load_inputs(inputs_given)
            else
              net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)
            end

            net.step_calc_forward
          end

          @mini_net_set.last.outputs_guessed
        end

        # TODO: utilize until_min_avg_error
        def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
          @mini_net_set.each_with_index do |net, index|
            index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@mini_net_set[index - 1].outputs_guessed)
            net.step_calc_forward
          end

          index_max = @mini_net_set.size - 1
          (0..index_max).to_a.reverse.each do |index|
            net = @mini_net_set[index]

            # index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@mini_net_set[index + 1].input_deltas[0..@mini_net_set[index + 1].height - 1])
            index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@mini_net_set[index + 1].input_deltas)

            net.step_calculate_error
            net.step_backpropagate
          end

          @mini_net_set.last.error_total
        end

        def guesses_best
          @mini_net_set.last.guesses_best
        end

        def step_calculate_error_distance_history
          @mini_net_set.last.step_calculate_error_distance_history
        end

        def error_distance_history
          @mini_net_set.last.error_distance_history
        end
      end
      # end
    end
  end
end
