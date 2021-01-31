require "json"
require "./error_stats.cr"
require "./breed_utils.cr"

module Ai4cr
  abstract class Breeder(T)
    # Implementaion example (taken from 'spec/ai4cr/breeder_spec.cr'):
    # ```
    # class MyClass
    #   # These are for breed relationship tracking:
    #   property birth_id : Int32
    #   property parent_a_id : Int32
    #   property parent_b_id : Int32
    #   property breed_delta : Float64
    #
    #   # These are to be set per child, but are
    #   #   NOT to be adjusted by the 'delta' value passed into the breeding process:
    #   #   (Add/Remove/Adjust for your particular class' needs.)
    #   property name : String
    #
    #   # These are to be adjusted by 'mix_parts' using the 'delta' value passed in:
    #   #   (Add/Remove/Adjust for your particular class' needs.)
    #   property some_value : Float64
    #
    #   include JSON::Serializable
    #
    #   def initialize(
    #     @name = "tbd",                # to be set per child
    #     @some_value : Float64? = -1.0 # to be adjusted by 'mix_parts(..)'
    #   )
    #     @birth_id = -1
    #     @parent_a_id = -1
    #     @parent_b_id = -1
    #     @breed_delta = 0.0
    #   end
    # end
    #
    # class MyBreeder < Ai4cr::Breeder(MyClass)
    #   def mix_parts(child : T, parent_a : T, parent_b : T, delta)
    #     child_part = mix_one_part(parent_a.some_value, parent_b.some_value, delta)
    #     child.some_value = child_part
    #
    #     child
    #   end
    # end
    # ```

    include JSON::Serializable
    include Ai4cr::BreedUtils

    def initialize
      # NOTE: We probably should convert the 'birth_id' from an instance variable to a class variable!
      #   Otherwise, you could get multiple instances with separate counters,
      #   which might or might not be desirable!
      @counter = SafeCounter.new
    end

    def create(**params)
      # i.e.: via NO parents
      birth_id = -1
      channel = Channel(Int32).new
      spawn do
        channel.send(@counter.inc(T.name))
      end
      birth_id = channel.receive

      child = T.new(**params)

      child.birth_id = birth_id

      child
    end

    def breed(parent_a : T, parent_b : T, delta = (rand*2 - 0.5), **params)
      # i.e.: VIA parents
      birth_id = -1
      channel = Channel(Int32).new
      spawn do
        channel.send(@counter.inc(T.name))
      end
      birth_id = channel.receive

      child = parts_to_copy(parent_a, parent_b, delta)

      child = mix_parts(child, parent_a, parent_b, delta)

      child.birth_id = birth_id
      child.parent_a_id = parent_a.birth_id
      child.parent_b_id = parent_b.birth_id
      child.breed_delta = delta

      child
    end

    def parts_to_copy(parent_a : T, parent_b : T, delta)
      # By default, we just copy everything from parent_a.
      # Since `self.clone` is erroring, we'll use from/to_json methods.
      T.from_json(parent_a.to_json)
    end

    # abstract
    def mix_parts(child : T, parent_a : T, parent_b : T, delta)
      # Sub-classes should do some sort of property mixing based on delta and both parents.
      # Typically, do something in sub-class's 'mix_one_part(..)' ...

      # And then be sure to return 'child'
      child
    end

    def mix_one_part(parent_a_part : Float64, parent_b_part : Float64, delta)
      vector_a_to_b = parent_b_part - parent_a_part
      parent_a_part + (delta * vector_a_to_b)
    end
  end
end
