module Ai4cr
  module Utils
    class Value
      LEVER_MAX = 1_000_000_000_000.0
      LEVEL_MIN = 0.000_000_000_000_1

      # ameba:disable Metrics/CyclomaticComplexity
      def self.protect_against_extremes(x)
        # How to avoid errors like:
        #   Unhandled exception in spawn: NaN not allowed in JSON (JSON::Error)
        #   from src/ai4cr/neural_network/cmn/mini_net_manager.cr:13:17 in 'copy_and_mix'
        #   from src/ai4cr/breed/manager.cr:108:17 in 'breed'
        #   from src/ai4cr/breed/manager.cr:274:30 in '->'
        #   from /home/drhuffman/.crenv/versions/0.36.0/share/crystal/src/primitives.cr:255:3 in 'run'
        #   from ???
        # Concept: If one 'lever' needs to move too big or too small of a distance,
        #   then force a different lever(s) to be pushed.

        # For any extreme reached, we might want to trigger some sort of 'pain':
        # TODO: What would represent 'pain' and how would we 'trigger' it?
        return -LEVER_MAX if x < -LEVER_MAX || -x == Float64::INFINITY
        return LEVER_MAX if x > LEVER_MAX || x == Float64::INFINITY

        # Although NAN isn't necessarilly an infinitely small number, we'll treat it as if it is.
        return -LEVEL_MIN if (x < -LEVEL_MIN && x <= 0.0) || -x == Float64::NAN
        return LEVEL_MIN if (x > LEVEL_MIN && x >= 0.0) || x == Float64::NAN

        # For no extreme reached, do not trigger any sort of 'pain':
        x
      end
      # ameba:enable Metrics/CyclomaticComplexity
    end
  end
end
