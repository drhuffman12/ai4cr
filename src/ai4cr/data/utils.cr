module Ai4cr
  module Data
    class Utils
      def self.rand_excluding(scale = 1, offset = 0.0, excludes = [0.0, 1.0], proximity = 0.0001)
        # Try to make sure that the random value is not within specified proximity of the excluded values.
        d = (rand * scale + offset)
        excludes.each do |ex|
          d = (rand * scale + offset) if (ex - d).abs < proximity
        end
        excludes.each do |ex|
          d = (rand * scale + offset) if (ex - d).abs < proximity
        end
        d
      end
    end
  end
end
