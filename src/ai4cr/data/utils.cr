module Ai4cr
  module Data
    class Utils
      # Ai4cr::Data::Utils.rand_excluding (with defaults), aka: def self.rand_zero_to_one_no_zero
      # -0.0..1.0 but no 0.0 and no 1.0
      def self.rand_excluding(scale = 1, offset = 0.0, excludes = [0.0, 1.0], proximity = 0.0001)
        d = rand_scaled_and_offset(scale, offset)

        # Try to make sure that the random value is not within specified proximity of the excluded values.
        # We could be more precise, but two loops like below should be sufficient.
        excludes.each do |ex|
          d = rand_scaled_and_offset(scale, offset) if (ex - d).abs < proximity
        end
        excludes.each do |ex|
          d = rand_scaled_and_offset(scale, offset) if (ex - d).abs < proximity
        end
        
        d
      end
      
      def self.rand_scaled_and_offset(scale = 1, offset = 0.0)
        rand * scale + offset
      end

      # Ai4cr::Data::Utils.rand_neg_one_to_pos_one_no_zero
      # -1.0..1.0 but no 0.0
      def self.rand_neg_one_to_pos_one_no_zero
        Ai4cr::Data::Utils.rand_excluding(scale: 2, offset: -1.0, excludes: [0.0])
      end

      # Ai4cr::Data::Utils.rand_neg_half_to_pos_one_and_half_no_zero_no_one
      # -0.5..1.5 but no 0.0 and no 1.0
      def self.rand_neg_half_to_pos_one_and_half_no_zero_no_one
        Ai4cr::Data::Utils.rand_excluding(scale: 2, offset: -0.5)
      end
    end
  end
end
