module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        # abstract class BreedParent
        module BreedParent
          
          # include Ai4cr::BreedParent

          # def initialize(name_suffix = "")
          #   super(name_suffix: name_suffix)
          # end

          def self.breed(
            parent_a : Ai4cr::NeuralNetwork::Cmn::MiniNet,
            parent_b : Ai4cr::NeuralNetwork::Cmn::MiniNet, 
            delta = (rand*2 - 0.5),
            name_suffix = ""
          )
            parent_c = parent_a.clone
            parent_c.name = init_name(name_suffix)

            parent_c.deriv_scale = breed_value(parent_a.deriv_scale, parent_b.deriv_scale, delta)
            parent_c.bias_default = breed_value(parent_a.bias_default, parent_b.bias_default, delta)
            parent_c.learning_rate = breed_value(parent_a.learning_rate, parent_b.learning_rate, delta)
            parent_c.momentum = breed_value(parent_a.momentum, parent_b.momentum, delta)

            parent_c.inputs_given = breed_nested(parent_a.inputs_given, parent_b.inputs_given, delta)
            parent_c.input_deltas = breed_nested(parent_a.input_deltas, parent_b.input_deltas, delta)

            parent_c.weights = breed_nested(parent_a.weights, parent_b.weights, delta)

            parent_c.outputs_guessed = breed_nested(parent_a.outputs_guessed, parent_b.outputs_guessed, delta)
            parent_c.output_deltas = breed_nested(parent_a.output_deltas, parent_b.output_deltas, delta)

            parent_c.last_changes = breed_nested(parent_a.last_changes, parent_b.last_changes, delta)

            parent_c.output_errors = breed_nested(parent_a.output_errors, parent_b.output_errors, delta)

            parent_c
          end

          def self.team(
            team_size,
            height, width,
            learning_style : LearningStyle = LS_RELU,

            # deriv_scale = rand / 2.0,

            disable_bias : Bool? = nil,
            # bias_default = 1.0,

            # learning_rate : Float64? = nil, momentum : Float64? = nil,
            history_size : Int32 = 10,
            name_suffix = "",
            training_round_qty = 1
          )
            member_config = {
              height: height,
              width: width,
              learning_style: learning_style,
              # deriv_scale: deriv_scale,
              disable_bias: disable_bias,
              # bias_default: bias_default,
              # learning_rate: learning_rate,
              # momentum: momentum,
              history_size: history_size,
              name_suffix: name_suffix,
            }
            Ai4cr::Team(Ai4cr::NeuralNetwork::Cmn::MiniNetConcerns::BreedParent).new(team_size, member_config, training_round_qty)
          end
        end
      end
    end
  end
end

# # TODO
# def breed_value(value_a, value_b, delta = rand*2)
#   #  Randomly pick a value somewhere between value_a and value_b or a bit on either side

#   direction = value_b.to_f - value_a.to_f
#   distance = rand* 2.0 - 0.5
#   value_a + direction * distance
# end

# child = breed_value(0,1)
# puts "child: #{child}"

# # parents = (-10..10).to_a.map{|i| i/10.0}
# # children = parents.map do |i|
# #   parents.map do |j|
# #     breed_value(i, j).round(1)
# #   end
# # end.flatten.sort

# # puts "parents: #{parents}"
# # puts "children: #{children.pretty_inspect}"
# # puts "children.tally: #{children.tally.pretty_inspect}"

# def breed_array()
# end

# def breed_nested(expected, real, delta = 0.01)
#   if expected.is_a?(Hash) && real.is_a?(Hash)
#     expected.keys.map do |key|
#       value_a = expected[key]
#       value_b = real[key]
#       { key => }
#     end
#   elsif expected.responds_to?(:each) && real.responds_to?(:each)
#     raise "Size Mismatch" unless expected.size == real.size

#     [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; breed_nested(ex, er, delta) }
#   else
#     breed_value expected, real, delta
#   end
# end

#   # def assert_equality_of_nested_list(expected, real)
#   def breed_nested(expected, real)
#     if expected.responds_to?(:each) && real.responds_to?(:each) && expected.size == real.size
#       [expected, real].transpose.each { |exer| ex = exer[0]; er = exer[1]; breed_nested(ex, er) }
#     else
#       # real.should eq(expected)
#       breed_nested(expected, real)
#     end
#   end

# parents1 = (-10..0).to_a.map{|i| i/10.0}
# parents2 = (0..10).to_a.map{|i| i/10.0}
# children = breed_nested(parents1,parents2)

# puts "parents1: #{parents1}"
# puts "parents2: #{parents2}"
# puts "children: #{children.pretty_inspect}"
# puts "children.tally: #{children.tally.pretty_inspect}"
