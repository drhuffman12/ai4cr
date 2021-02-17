module Ai4cr
  module Breed
    class StructureError < ArgumentError; end
    
    abstract class Manager(T)
      # class MyCounter < Counter::Safe; end

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
      #   ALLOWED_STRING_LAST  = "z" # 'z' # .ord
      #   ALLOWED_STRINGS      = (ALLOWED_STRING_FIRST..ALLOWED_STRING_LAST).to_a
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

      ############################################################################
      # TODO: WHY is this required?
      # NOTE: Sub-classes MUST include the following two lines:
      include JSON::Serializable
      class_getter counter : CounterSafe::Exclusive = CounterSafe::Exclusive.new

      def initialize; end

      ############################################################################

      #       property team_size : Int32 = 2
      #       property training_round_qty : Int32 = 10
      #       property training_round_indexes = Array(Int32).new
      #       # getter team_indexes : Array(Int32)
      #       getter team_members : Array(T) # .new

      #       property team_last_id : Int32
      #       property manager = Manager(T).new

      #       # property member_config
      # include Ai4cr::Breed::Utils

      # def initialize
      #   # NOTE: We probably should convert the 'birth_id' from an instance variable to a class variable!
      #   #   Otherwise, you could get multiple instances with separate counters,
      #   #   which might or might not be desirable!
      #   # @@counter = init_counter
      # end

      # def init_counter
      #   @@counter = CounterSafe::Exclusive.new
      # end

      def counter
        @@counter
      end

      def counter_reset(value = 0)
        @@counter.reset(T.name, value)
      end

      def create(**params)
        # i.e.: via NO parents
        channel = Channel(Int32).new
        spawn do
          value = @@counter.inc(T.name)
          channel.send(value)
        end
        birth_id = channel.receive

        child = T.new(**params)

        child.birth_id = birth_id

        child
      end

      def estimate_better_delta(error_a : Float64, error_b : Float64)
        # An error value of '0.0' is when you're at the soultion.
        # The error values are assumed to be positive (i.e.: radius from solution), but could be negative.
        # So, the solution should be where the two errors overlap.
        # Of course if the solution is not along the line between 'a' and 'b',
        #   then you'll need to diverge from that line.

        vector_a_to_b = error_b - error_a
        # zero = error_a + delta * vector_a_to_b
        # so (avoid div by 0 and then) return ...
        vector_a_to_b == 0.0 ? 0.0 : -error_a / vector_a_to_b
      end

      def breed(parent_a : T, parent_b : T, delta = Ai4cr::Data::Utils.rand_excluding(scale: 2, offset: -0.5))
        breed_validations(parent_a, parent_b, delta)

        # i.e.: VIA parents
        birth_id = breed_counter_tick
        child = copy_and_mix(parent_a, parent_b, delta)
        child = breed_id_and_delta(child, birth_id, parent_a, parent_b, delta)
        child.error_stats = Ai4cr::ErrorStats.new(parent_a.error_stats.history_size)

        child
      end

      def breed_validations(parent_a : T, parent_b : T, delta)
        raise "Parents must be Breed Clients!" unless T < Breed::Client
      end

      def breed_counter_tick
        channel = Channel(Int32).new
        spawn do
          channel.send(@@counter.inc(T.name))
        end
        channel.receive
      end

      def breed_id_and_delta(child, birth_id, parent_a, parent_b, delta)
        child.birth_id = birth_id
        child.parent_a_id = parent_a.birth_id
        child.parent_b_id = parent_b.birth_id
        child.breed_delta = delta

        child
      end

      def copy_and_mix(parent_a, parent_b, delta)
        child = parts_to_copy(parent_a, parent_b, delta)
        mix_parts(child, parent_a, parent_b, delta)
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

        # some_value = mix_one_part_number(parent_a.some_value, parent_b.some_value, delta)
        # child.some_value = some_value

        # some_array = mix_nested_parts(parent_a.some_array, parent_b.some_array, delta)
        # child.some_array = some_array

        # some_string = mix_nested_parts(parent_a.some_string, parent_b.some_string, delta)
        # child.some_string = some_string

        # And then be sure to return 'child'
        child
      end

      def mix_one_part_number(parent_a_part : Number, parent_b_part : Number, delta)
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
          raise "Unhandled values; parent_a_part, parent_b_part == #{[parent_a_part, parent_b_part]}, classes: #{[parent_a_part.class, parent_b_part.class]}"
        end
      end
    end
  end
end
