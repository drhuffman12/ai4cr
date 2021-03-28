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

      def clone
        raise "TO BE IMPLEMENTED"
      end

      def error_hist_stats(in_bw = false)
        ehs = "'"
        begin
          ehs += birth_id.to_s
          ehs += " "
          ehs += name.to_s
          ehs += " => "
          ehs += (error_stats.plot_error_distance_history(in_bw: in_bw)).to_s
          ehs += " @ "
          ehs += (error_stats.score).to_s
        rescue ex
          ehs += "(ERROR: #{ex.class} re '#{ex.message}' at [#{ex.backtrace.join("\n")}])"
        end
        ehs += "'"
        ehs
      end
    end
  end
end
