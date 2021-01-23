require "json"
require "./error_stats.cr"
require "./breed_utils.cr"

module Ai4cr
  module BreedParent # (T)
  # abstract class BreedParent
    # getter name : String
    # getter error_stats = Ai4cr::ErrorStats.new

    include Ai4cr::BreedUtils

    # def initialize(name_suffix = "")
    #   @name = init_name(name_suffix)
    # end

    def init_name(name_suffix : String)
      {
        klass:  self.class.name,
        ms:     Time.utc.to_unix_ms,
        suffix: name_suffix,
      }.to_json
    end

    # def 

    def breed_with(parent_b, delta = (rand*2 - 0.5),
      name_suffix = "")
      raise "TO BE IMPLEMENTED IN REAL CLASS"
      # self.clone
      # breed_non_changing_values(my_clone : T, other_parent : T, delta)
    end

    # def breed(parent_b, delta = (rand*2 - 0.5),
    #   name_suffix = "")
    #   raise "TO BE IMPLEMENTED IN REAL CLASS"
    #   # self.clone
    #   # breed_non_changing_values(my_clone : T, other_parent : T, delta)
    # end

    # def self.train_next_gen(next_gen : Array(T), inputs_given, outputs_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT))
    #   raise "TO BE IMPLEMENTED IN REAL CLASS"
    #   # self.clone
    #   # breed_non_changing_values(my_clone : T, other_parent : T, delta)
    # end

    # def self.breed_non_changing_values(my_clone : T, other_parent : T, delta) : T
    # end
  end
end
