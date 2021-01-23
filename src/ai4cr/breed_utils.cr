module Ai4cr
  module BreedUtils
    def breed_value(value_a, value_b, delta = (rand*2 - 0.5))
      # Randomly pick a variant of two numbers by picking
      #   a value w/in 50% of value_a or value_b along the
      #   line that intersects value_a and value_b.

      direction = 1.0 * (value_b - value_a)
      distance = delta.is_a?(Number) ? delta.to_f : rand*2.0 - 0.5
      (value_a + direction * distance)
    end

    def breed_nested(value_a, value_b, delta = (rand*2 - 0.5))
      # case
      # when value_a.responds_to?(:keys) && value_b.responds_to?(:keys)
      # TODO: Find a fix for other datatypes, such as a Hash(T,U)
      #   # puts "Hash, value_a, value_b == #{[value_a, value_b]}"
      #   value_c = value_a.clone
      #   value_c.keys.each do |key|
      #     a = value_a[key]
      #     b = value_b[key]
      #     # [ key, breed_nested(a, b, delta) ]
      #     value_c[key] = breed_nested(a, b, delta)
      #   end
      #   value_c

      case
      when value_a.is_a?(Number) && value_a.is_a?(Number)
        breed_value(value_a, value_b, delta)
      when value_a.responds_to?(:each) && value_b.responds_to?(:each) && value_a.size == value_b.size
        [value_a, value_b].transpose.map { |tran| va = tran[0]; vb = tran[1]; breed_nested(va, vb, delta) }
      else
        raise "Unhandled values; value_a, value_b == #{[value_a, value_b]}"
      end
    end
  end
end
