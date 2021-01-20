require "json"
require "./error_stats.cr"

# include Ai4cr::BreedParent(T)
module Ai4cr
  module BreedParent(T)
    getter name : String
    getter error_stats = Ai4cr::ErrorStats.new

    # include Ai4cr::BreedUtils

    def init_name(name_suffix)
      {
        klass:  self.class.name,
        ms:     Time.utc.to_unix_ms,
        suffix: name_suffix,
      }.to_json
    end

    def breed(parent_a : T, parent_b : T, delta = (rand*2 - 0.5))
      raise "TO BE IMPLEMENTED IN REAL CLASS"
      # self.clone
      # breed_non_changing_values(my_clone : T, other_parent : T, delta)
    end

    # def self.breed_non_changing_values(my_clone : T, other_parent : T, delta) : T
    # end
  end
end
