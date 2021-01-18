module Ai4cr
  module BreedUtils
    def breed_value(value_a, value_b, delta = rand*2)
      #  Randomly pick a value somewhere between value_a and value_b or a bit on either side

      direction = 1.0 * (value_b - value_a)
      distance = rand*2.0 - 0.5
      (value_a + direction * distance) # .as_a(value_b.class)
    end

    # def assert_equality_of_nested_list(value_a, value_b)
    def breed_nested(value_a, value_b)
      case
      when value_a.is_a?(Number) && value_a.is_a?(Number)
        # puts "Float64, value_a, value_b == #{[value_a, value_b]}"
        breed_value(value_a, value_b)
        # when value_a.responds_to?(:keys) && value_b.responds_to?(:keys)
        # TODO: Find a fix for other datatypes, such as a Hash(T,U)
        #   puts "Hash, value_a, value_b == #{[value_a, value_b]}"
        #   value_c = value_a.clone
        #   value_c.keys.each do |key|
        #     a = value_a[key]
        #     b = value_b[key]
        #     # [ key, breed_nested(a, b) ]
        #     value_c[key] = breed_nested(a, b)
        #   end
        #   value_c
      when value_a.responds_to?(:each) && value_b.responds_to?(:each) && value_a.size == value_b.size
        # puts "Array(...), value_a, value_b == #{[value_a, value_b]}"
        [value_a, value_b].transpose.map { |exer| ex = exer[0]; er = exer[1]; breed_nested(ex, er) }
      else
        # puts "else, value_a, value_b == #{[value_a, value_b]}"
        raise "value_a, value_b == #{[value_a, value_b]}"
      end
    end
  end
end
