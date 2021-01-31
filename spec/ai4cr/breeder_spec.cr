require "./../spec_helper"
require "./../spectator_helper"

# class MyClass
#   property birth_id : Int32
#   property parent_a_id : Int32
#   property parent_b_id : Int32
#   property breed_delta : Float64

#   property some_value : Float64
#   property name : String

#   include JSON::Serializable

#   def initialize(
#     @birth_id = -1,
#     @name = "tbd",
#     @some_value : Float64? = -1.0
#   )
#     @parent_a_id = -1
#     @parent_b_id = -1
#     @breed_delta = 0.0
#   end
# end

class MyClass
  # These are for breed relationship tracking:
  property birth_id : Int32
  property parent_a_id : Int32
  property parent_b_id : Int32
  property breed_delta : Float64

  # These are to be set per child, but are
  #   NOT to be adjusted by the 'delta' value passed into the breeding process:
  #   (Add/Remove/Adjust for your particular class' needs.)
  property name : String

  # These are to be adjusted by the 'delta' value passed into 'mix_parts':
  #   (Add/Remove/Adjust for your particular class' needs.)
  property some_value : Float64

  include JSON::Serializable

  def initialize(
    @birth_id = -1,
    @name = "tbd",                # to be set per child
    @some_value : Float64? = -1.0 # to be adjusted by 'mix_parts(..)'
  )
    @parent_a_id = -1
    @parent_b_id = -1
    @breed_delta = 0.0
  end
end

class MyBreeder < Ai4cr::Breeder(MyClass)
  def mix_parts(child : T, parent_a : T, parent_b : T, delta)
    child_part = mix_one_part(parent_a.some_value, parent_b.some_value, delta)
    child.some_value = child_part

    child
  end
end

Spectator.describe Ai4cr::Breeder do
  let(my_breeder) { MyBreeder.new }
  let(delta_child_1) { (rand*2 - 0.5) }
  let(delta_child_2) { (rand*2 - 0.5) }

  let(ancestor_adam_value) { 0.0 }
  let(ancestor_eve_value) { 1.0 }

  let(ancestor_adam) { my_breeder.create(name: "Adam", some_value: ancestor_adam_value) }
  let(ancestor_eve) { my_breeder.create(name: "Eve", some_value: ancestor_eve_value) }

  describe "#initialize" do
    context "For Adam and Eve examples" do
      it "birth_id's are in the correct order (when birthed in correct order" do
        birth_counter = 0
        puts
        puts "ancestor_adam: #{ancestor_adam.to_json}"
        puts "ancestor_eve: #{ancestor_eve.to_json}"

        expect(ancestor_adam.birth_id).to eq(birth_counter += 1)
        expect(ancestor_adam.some_value).to eq(ancestor_adam_value)

        expect(ancestor_eve.birth_id).to eq(birth_counter += 1)
        expect(ancestor_eve.some_value).to eq(ancestor_eve_value)

        # cain
        child_1 = my_breeder.breed(ancestor_adam, ancestor_eve, delta: delta_child_1)
        child_1.name = "Cain, child of #{child_1.name} and #{ancestor_eve.name}"

        puts "child_1: #{child_1.to_json}"
        expect(child_1.birth_id).to eq(birth_counter += 1)
        expect(child_1.some_value).to eq(delta_child_1)

        # abel

        child_2 = my_breeder.breed(ancestor_adam, ancestor_eve, delta: delta_child_2)
        child_2.name = "Abel, child of #{child_2.name} and #{ancestor_eve.name}"

        puts "child_2: #{child_2.to_json}"
        expect(child_2.birth_id).to eq(birth_counter += 1)
        expect(child_2.some_value).to eq(delta_child_2)

        puts
        puts "Now, in order or youngest to oldest:"
        [ancestor_adam, ancestor_eve, child_1, child_2].sort_by do |person|
          -person.birth_id
        end.each do |person|
          puts "person: #{person.to_json}"
        end
      end
    end
  end
end
