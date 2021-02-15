module Ai4cr
  module Breed
    module Client
      # These are for breed relationship tracking:
      property birth_id : Int32 = -1
      property name : String = Time.utc.to_s
      property parent_a_id : Int32 = -1
      property parent_b_id : Int32 = -1
      property breed_delta : Float64 = 0.0
      property error_stats = Ai4cr::ErrorStats.new

      def history_size
        error_stats.history_size
      end
    end
  end
end
