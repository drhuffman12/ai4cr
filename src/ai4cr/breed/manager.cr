require "json"

# require "./breed_utils.cr"

module Ai4cr
  module Breed
    abstract class Manager(T)
      # Implementaion example (taken from 'spec/ai4cr/breed/manager_spec.cr'):
      # ```
      # class MyBreed
      #   include JSON::Serializable
      #   include Ai4cr::Breed::Client
      #
      #   # These are to be set per child, but are
      #   #   NOT to be adjusted by the 'delta' value passed into the breeding process:
      #   #   (Add/Remove/Adjust for your particular class' needs.)
      #   property name : String = "tbd"
      #
      #   # These are to be adjusted by the 'delta' value passed into 'mix_parts':
      #   #   (Add/Remove/Adjust for your particular class' needs.)
      #   property some_value : Float64 = -1.0
      #   property some_array = Array(Float64).new(2) { rand }
      #
      #   ALLOWED_STRING_FIRST = "a" # 'a' # .ord
      #   ALLOWED_STRING_LAST = "z" # 'z' # .ord
      #   ALLOWED_STRINGS = (ALLOWED_STRING_FIRST..ALLOWED_STRING_LAST).to_a
      #   property some_string : String = (ALLOWED_STRINGS.sample) * 2
      #
      #   def initialize(@name, @some_value)
      #   end
      # end
      #
      # class MyBreeder < Ai4cr::Breed::Manager(MyBreed)
      #   def mix_parts(child : T, parent_a : T, parent_b : T, delta)
      #     some_value = mix_one_part_number(parent_a.some_value, parent_b.some_value, delta)
      #     child.some_value = some_value
      #
      #     some_array = mix_nested_parts(parent_a.some_array, parent_b.some_array, delta)
      #     child.some_array = some_array
      #
      #     some_string = mix_nested_parts(parent_a.some_string, parent_b.some_string, delta)
      #     child.some_string = some_string
      #
      #     child
      #   end
      # end
      # ```

      include JSON::Serializable

      # include Ai4cr::Breed::Utils

      def initialize
        # NOTE: We probably should convert the 'birth_id' from an instance variable to a class variable!
        #   Otherwise, you could get multiple instances with separate counters,
        #   which might or might not be desirable!
        @counter = SafeCounter.new
      end

      def create(**params)
        # i.e.: via NO parents
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
        raise "Must be a Breed Client!" unless T < Breed::Client
        
        # i.e.: VIA parents
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
        # Typically, do something in sub-class's 'mix_one_part_number(..)' ...

        # And then be sure to return 'child'
        child
      end

      def mix_one_part_number(parent_a_part : Float64, parent_b_part : Float64, delta)
        vector_a_to_b = parent_b_part - parent_a_part
        parent_a_part + (delta * vector_a_to_b)
      end

      def mix_one_part_string(parent_a_part : String, parent_b_part : String, delta)
        # TODO: Add code/classes to verify
        # NOTE: Sub-classes might want to adjust the logic for this
        delta < 0.5 ? parent_a_part : parent_b_part
        # delta < rand ? parent_a_part : parent_b_part
      end

      def mix_nested_parts(parent_a_part, parent_b_part, delta)
        # TODO: Expand to handle other nested data types, such as
        # * hashes
        # * characters
        case
        when parent_a_part.is_a?(String) && parent_b_part.is_a?(String)
          mix_one_part_string(parent_a_part, parent_b_part, delta)
        when parent_a_part.is_a?(Number) && parent_b_part.is_a?(Number)
          mix_one_part_number(parent_a_part, parent_b_part, delta)
        when parent_a_part.responds_to?(:each) && parent_b_part.responds_to?(:each) && parent_a_part.size == parent_b_part.size
          # NOTE: This works for arrays, but not hashes.
          [parent_a_part, parent_b_part].transpose.map { |tran| va = tran[0]; vb = tran[1]; mix_nested_parts(va, vb, delta) }
        else
          raise "Unhandled values; parent_a_part, parent_b_part == #{[parent_a_part, parent_b_part]}"
        end
      end
    end
  end
end
