require "./../../spec_helper"
require "./../../spectator_helper"

class MyBreed
  include JSON::Serializable
  include Ai4cr::Breed::Client

  # These are to be set per child, but are
  #   NOT to be adjusted by the 'delta' value passed into the breeding process:
  #   (Add/Remove/Adjust for your particular class' needs.)
  property name : String = "tbd"

  # These are to be adjusted by the 'delta' value passed into 'mix_parts':
  #   (Add/Remove/Adjust for your particular class' needs.)
  property some_value : Float64 = -1.0
  property some_array = Array(Float64).new(2) { rand }

  ALLOWED_STRING_FIRST = "a"
  ALLOWED_STRING_LAST  = "z"
  ALLOWED_STRINGS      = (ALLOWED_STRING_FIRST..ALLOWED_STRING_LAST).to_a
  property some_string : String = (ALLOWED_STRINGS.sample) * 2

  def initialize(@name, @some_value)
  end
end

class MyBreedManager < Ai4cr::Breed::Manager(MyBreed)
  def mix_parts(child : T, parent_a : T, parent_b : T, delta)
    some_value = mix_one_part_number(parent_a.some_value, parent_b.some_value, delta)
    child.some_value = some_value

    some_array = mix_nested_parts(parent_a.some_array, parent_b.some_array, delta)
    child.some_array = some_array

    some_string = mix_nested_parts(parent_a.some_string, parent_b.some_string, delta)
    child.some_string = some_string

    child
  end
end

def puts_debug(message = "")
  puts message if ENV.has_key?("DEBUG") && ENV["DEBUG"] == "1"
end

Spectator.describe Ai4cr::Breed::Manager do
  let(my_breed_manager) { MyBreedManager.new }

  describe "For Adam and Eve examples" do
    # TODO: Split this up into smaller tests!
    let(delta_child_1) { (rand*2 - 0.5) }
    let(delta_child_2) { (rand*2 - 0.5) }

    let(ancestor_adam_value) { 0.0 }
    let(ancestor_eve_value) { 1.0 }

    let(ancestor_adam) { my_breed_manager.create(name: "Adam", some_value: ancestor_adam_value) }
    let(ancestor_eve) { my_breed_manager.create(name: "Eve", some_value: ancestor_eve_value) }

    let(some_array_expected_1) {
      ancestor_adam.some_array.map_with_index do |sa, i|
        parent_a_part = sa
        parent_b_part = ancestor_eve.some_array[i]

        vector_a_to_b = parent_b_part - parent_a_part
        parent_a_part + (delta_child_1 * vector_a_to_b)
      end
    }
    let(some_array_expected_2) {
      ancestor_adam.some_array.map_with_index do |sa, i|
        parent_a_part = sa
        parent_b_part = ancestor_eve.some_array[i]

        vector_a_to_b = parent_b_part - parent_a_part
        parent_a_part + (delta_child_2 * vector_a_to_b)
      end
    }

    let(some_string_expected_1) {
      parent_a_part = ancestor_adam.some_string
      parent_b_part = ancestor_eve.some_string

      delta_child_1 < 0.5 ? parent_a_part : parent_b_part
    }
    let(some_string_expected_2) {
      parent_a_part = ancestor_adam.some_string
      parent_b_part = ancestor_eve.some_string

      delta_child_2 < 0.5 ? parent_a_part : parent_b_part
    }

    it "birth_id's are in the consistent order (when birthed in order" do
      expected_birth_counter = 0
      puts_debug
      puts_debug "ancestor_adam: #{ancestor_adam.to_json}"
      puts_debug
      puts_debug "ancestor_eve: #{ancestor_eve.to_json}"

      expected_birth_counter += 1

      expect(ancestor_adam.birth_id).to eq(expected_birth_counter)
      expect(ancestor_adam.some_value).to eq(ancestor_adam_value)

      expected_birth_counter += 1
      expect(ancestor_eve.birth_id).to eq(expected_birth_counter)
      expect(ancestor_eve.some_value).to eq(ancestor_eve_value)

      # cain
      child_1 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
      child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"

      puts_debug "child_1: #{child_1.to_json}"
      expected_birth_counter += 1
      expect(child_1.birth_id).to eq(expected_birth_counter)
      expect(child_1.some_value).to eq(delta_child_1)

      expect(child_1.some_array).to eq(some_array_expected_1)
      expect(child_1.some_string).to eq(some_string_expected_1)

      # abel
      child_2 = my_breed_manager.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
      child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"

      puts_debug "child_2: #{child_2.to_json}"
      expected_birth_counter += 1
      expect(child_2.birth_id).to eq(expected_birth_counter)
      expect(child_2.some_value).to eq(delta_child_2)

      expect(child_2.some_array).to eq(some_array_expected_2)
      expect(child_2.some_string).to eq(some_string_expected_2)

      puts_debug
      puts_debug "Now, in order or youngest to oldest:"
      [ancestor_adam, ancestor_eve, child_1, child_2].sort_by do |person|
        (-person.birth_id)
      end.each do |person|
        puts_debug "person: #{person.to_json}"
      end
    end
  end
end
