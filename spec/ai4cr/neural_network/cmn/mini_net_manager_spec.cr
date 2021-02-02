require "./../../../spec_helper"
require "./../../../spectator_helper"

def puts_debug(message = "")
  puts message if ENV.has_key?("DEBUG") && ENV["DEBUG"] == "1"
end

def rand_excluding
  # TODO: make this more generic for init'ing of various net values (where we want to avoid zero's and ones)
  excludes = [0.0, 1.0]
  # Try to make sure that the random value is not one of the excluded values.
  # Below isn't a guarantee, but should be good enough for the tests.
  d = (rand * 2 - 0.5)
  excludes.each do |ex|
    d = (rand * 2 - 0.5) if (ex - d).abs < 0.0001
  end
  excludes.each do |ex|
    d = (rand * 2 - 0.5) if (ex - d).abs < 0.0001
  end
  d
end

Spectator.describe Ai4cr::NeuralNetwork::Cmn::MiniNetManager do
  let(my_breed_manager) { Ai4cr::NeuralNetwork::Cmn::MiniNetManager.new }

  describe "For Adam and Eve examples" do
    let(delta_child_1) { rand_excluding }
    let(delta_child_2) { rand_excluding }

    let(ancestor_adam_learning_rate_expected) { 0.1 }
    let(ancestor_eve_learning_rate_expected) { 0.9 }

    let(config_default_randomized) {
      Ai4cr::NeuralNetwork::Cmn::MiniNet.new.config
    }

    let(config_adam) {
      config_default_randomized.merge(
        name: "Adam", learning_rate: ancestor_adam_learning_rate_expected
      )
    }

    let(config_eve) {
      config_default_randomized.merge(
        name: "Eve", learning_rate: ancestor_eve_learning_rate_expected
      )
    }

    let(ancestor_adam) { my_breed_manager.create(**config_adam) }
    let(ancestor_eve) { my_breed_manager.create(**config_eve) }

    before_each do
      my_breed_manager.counter_reset
    end

    it "learning rates of Adam and Eve are different" do
      expect(ancestor_adam.learning_rate).to eq(ancestor_adam_learning_rate_expected)
      expect(ancestor_eve.learning_rate).to eq(ancestor_eve_learning_rate_expected)
      expect(ancestor_adam.learning_rate).not_to eq(ancestor_eve.learning_rate)
    end

    # context "birth_id's are in the consistent order (when birthed in order" do
    #   it "first Adam" do
    #     expected_birth_counter = 0

    #     # Adam
    #     expected_birth_counter += 1
    #     expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

    #     puts_debug
    #     puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
    #   end

    #   it "first Adam then Eve" do
    #     expected_birth_counter = 0

    #     # Adam
    #     expected_birth_counter += 1
    #     expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

    #     # Eve
    #     expected_birth_counter += 1
    #     expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

    #     puts_debug
    #     puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
    #     puts_debug
    #     puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
    #   end

    #   it "first Adam then Eve followed by Cain" do
    #     expected_birth_counter = 0

    #     # Adam
    #     expected_birth_counter += 1
    #     expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

    #     # Eve
    #     expected_birth_counter += 1
    #     expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

    #     # Cain
    #     expected_birth_counter += 1
    #     child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
    #     child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
    #     expect(child_1.birth_id).to eq(expected_birth_counter)

    #     puts_debug
    #     puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
    #     puts_debug
    #     puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
    #     puts_debug
    #     puts_debug "child_1: #{child_1.to_json}"
    #   end

    #   it "first Adam then Eve followed by Cain and then Abel" do
    #     expected_birth_counter = 0

    #     # Adam
    #     expected_birth_counter += 1
    #     expect(ancestor_adam.birth_id).to eq(expected_birth_counter)

    #     # Eve
    #     expected_birth_counter += 1
    #     expect(ancestor_eve.birth_id).to eq(expected_birth_counter)

    #     # Cain
    #     expected_birth_counter += 1
    #     child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
    #     child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
    #     expect(child_1.birth_id).to eq(expected_birth_counter)

    #     # Abel
    #     expected_birth_counter += 1
    #     child_2 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
    #     child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"
    #     expect(child_2.birth_id).to eq(expected_birth_counter)

    #     puts_debug
    #     puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
    #     puts_debug
    #     puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
    #     puts_debug
    #     puts_debug "child_1: #{child_1.to_json}"
    #     puts_debug
    #     puts_debug "child_2: #{child_2.to_json}"
    #   end
    # end

    describe "#mix_parts" do
      context "first Adam then Eve followed by Cain" do
        let(child_1) { my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1) }

        let(learning_rate_expected_1) {
          parent_a_part = ancestor_adam.learning_rate
          parent_b_part = ancestor_eve.learning_rate

          vector_a_to_b = parent_b_part - parent_a_part
          parent_a_part + (delta_child_1 * vector_a_to_b)
        }

        let(momentum_expected_1) {
          parent_a_part = ancestor_adam.momentum
          parent_b_part = ancestor_eve.momentum

          vector_a_to_b = parent_b_part - parent_a_part
          parent_a_part + (delta_child_1 * vector_a_to_b)
        }

        before_each do
          # Force variable to be initialized in specific order

          # Adam
          ancestor_adam.name = ancestor_adam.name + ""

          # Eve
          ancestor_eve.name = ancestor_eve.name + ""

          # Cain
          child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"
        end

        it "debug" do
          puts_debug
          puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
          puts_debug
          puts_debug "ancestor_eve: #{ancestor_eve.to_json}"
          puts_debug
          puts_debug "child_1: #{child_1.to_json}"
          puts_debug

          # puts_debug
          # puts_debug "ancestor_adam: #{ancestor_adam.to_pretty_json}"
          # puts_debug
          # puts_debug "ancestor_eve: #{ancestor_eve.to_pretty_json}"
          # puts_debug
          # puts_debug "child_1: #{child_1.to_pretty_json}"
          # puts_debug
        end

        context "first child" do
          context "gets expected value(s) for" do
            it "learning_rate" do
              expect(child_1.learning_rate).to eq(learning_rate_expected_1)
            end

            it "momentum" do
              expect(child_1.momentum).to eq(momentum_expected_1)
            end

            # ... do likewise for other applicable variables
          end

          context "does not get exact copy of either parent for values of" do
            it "learning_rate" do
              expect(child_1.learning_rate).not_to eq(ancestor_adam)
              expect(child_1.learning_rate).not_to eq(ancestor_eve)
            end

            it "momentum" do
              expect(child_1.momentum).not_to eq(ancestor_adam)
              expect(child_1.momentum).not_to eq(ancestor_eve)
            end

            # ... do likewise for other applicable variables
          end
        end
      end
    end
  end
end

# let(weights_expected_1) {
#   ancestor_adam.weights.map_with_index do |sa, i|
#     parent_a_part = sa
#     parent_b_part = ancestor_eve.weights[i]

#     vector_a_to_b = parent_b_part - parent_a_part
#     parent_a_part + (delta_child_1 * vector_a_to_b)
#   end
# }
# let(weights_expected_2) {
#   ancestor_adam.weights.map_with_index do |sa, i|
#     parent_a_part = sa
#     parent_b_part = ancestor_eve.weights[i]

#     vector_a_to_b = parent_b_part - parent_a_part
#     parent_a_part + (delta_child_2 * vector_a_to_b)
#   end
# }

# let(momentum_expected_1) {
#   ancestor_adam.momentum.map_with_index do |sa, i|
#     parent_a_part = sa
#     parent_b_part = ancestor_eve.momentum[i]

#     vector_a_to_b = parent_b_part - parent_a_part
#     parent_a_part + (delta_child_1 * vector_a_to_b)
#   end
# }
# let(momentum_expected_2) {
#   ancestor_adam.momentum.map_with_index do |sa, i|
#     parent_a_part = sa
#     parent_b_part = ancestor_eve.momentum[i]

#     vector_a_to_b = parent_b_part - parent_a_part
#     parent_a_part + (delta_child_2 * vector_a_to_b)
#   end
# }

# let(config_cain) {
#   config_default_randomized.merge(
#     name: "Cain", learning_rate: ancestor_adam_learning_rate_expected
#   )
# }

# let(config_abel) {
#   config_default_randomized.merge(
#     name: "Abel", learning_rate: ancestor_adam_learning_rate_expected
#   )
# }

# expect(ancestor_adam.learning_rate).to eq(ancestor_adam_learning_rate_expected)

# expected_birth_counter += 1
# expect(ancestor_eve.birth_id).to eq(expected_birth_counter)
# expect(ancestor_eve.learning_rate).to eq(ancestor_eve_learning_rate_expected)

# # cain
# child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
# child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"

# puts_debug "child_1: #{child_1.to_json}"
# expected_birth_counter += 1
# expect(child_1.birth_id).to eq(expected_birth_counter)
# expect(child_1.learning_rate).to eq(delta_child_1)

# # expect(child_1.weights).to eq(weights_expected_1)
# # expect(child_1.momentum).to eq(momentum_expected_1)

# # abel
# child_2 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
# child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"

# puts_debug "child_2: #{child_2.to_json}"
# expected_birth_counter += 1
# expect(child_2.birth_id).to eq(expected_birth_counter)
# expect(child_2.learning_rate).to eq(delta_child_2)

# # expect(child_2.weights).to eq(weights_expected_2)
# # expect(child_2.momentum).to eq(momentum_expected_2)

# puts_debug
# puts_debug "Now, in order or youngest to oldest:"
# [ancestor_adam, ancestor_eve, child_1, child_2].sort_by do |person|
#   (- person.birth_id)
# end.each do |person|
#   puts_debug "person: #{person.to_json}"
# end
