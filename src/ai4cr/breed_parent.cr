require "json"
require "./error_stats.cr"
require "./breed_utils.cr"

module Ai4cr
  class BreedMismatch < Exception
    getter parent_a_class_name : String
    getter parent_b_class_name : String
    def initialize(parent_a, parent_b) # , @message : String? = nil, @cause : Exception? = nil)
      @parent_a_class_name = parent_a.class.name
      @parent_b_class_name = parent_b.class.name

      super(message: class_miss_match_error_message) # , cause: cause)
    end

    def class_miss_match_error_message
      "Breed Class Mis-match; parent_a.class.name: #{parent_a_class_name},  parent_b.class.name: #{parent_b_class_name}"
    end
  end

  module BreedParent
    # Classes that implement 'Ai4cr::BreedParent' must do at least the following:
    # * getter for 'name'
    # * include JSON::Serializable
    # * include Ai4cr::BreedParent
    # * init 'name' in the contructor
    #
    # For example:
    # ```
    # class Bp1
    #   getter name : String 
    #
    #   include JSON::Serializable
    #   include Ai4cr::BreedParent 
    #
    #   def initialize(name_suffix = "one") # Time.utc.to_s
    #     @name = init_name(name_suffix)
    #   end
    # end
    # ```
    #
    # Additionally, the `#parts_to_copy(..)` and `#mix_parts(..)` methods should be implemented
    
    include Ai4cr::BreedUtils

    # getter amount_vector = 0.0
    getter amount_a : Float64 = 1.0
    getter amount_b : Float64 = 0.0

    def init_name(name_suffix : String)
      {
        klass:  self.class.name,
        ms:     Time.utc.to_unix_ms,
        suffix: name_suffix,
      }.to_json
    end

    def breed_with(parent_b, delta = (rand*2 - 0.5), name_suffix = "")
      check_partner_class(self, parent_b)

      child = parts_to_copy(parent_b)
      mix_parts(child, parent_b, delta)
    end

    protected def calc_amounts(delta)
      @amount_a = 1.0 - delta
      @amount_b = delta
    end

    protected def check_partner_class(parent_a, parent_b)
      return if parent_a.class == parent_b.class
      raise Ai4cr::BreedMismatch.new(parent_a, parent_b)
    end

    protected def parts_to_copy(parent_b)
      # By default, we just copy everything from parent_a.
      # Since `self.clone` is erroring, we'll use from/to_json methods.
      # Hence, `include JSON::Serializable` is required in parent methods
      self.class.from_json(self.to_json)
    end

    protected def mix_parts(child, parent_b, delta)
      # Instead of returning 'child', sub-classes should do some sort of property mixing based on delta and both parents
      child
    end

    protected def mix_one_part(parent_a_part, parent_b_part, delta)

    end
  end
end
