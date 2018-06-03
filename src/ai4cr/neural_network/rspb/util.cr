module Ai4cr
  module NeuralNetwork
    module Rspb
      # Ai4cr::NeuralNetwork::Math
      struct Util
        # Generates an array of prime-like positive and negative offsets (with -1, 0, and 1 included)
        def self.gen_prime_offsets(prime_qty)
          range = (0..prime_qty-1)
          x_prev = 1
          x_cur = 2
          x_next = 0
          offsets = range.map do |i|
            x_next = x_prev + x_cur
            x = x_prev
            x_prev = x_cur
            x_cur = x_next
            x
          end
          (offsets.reverse.map { |o| -o } + [0] + offsets).flatten
        end

        def self.gen_zoom_scales(zoom_qty)
          range = (0..zoom_qty-1)
          range.map do |i|
            2 ** i
          end
        end

        def self.gen_zoomed_prime_offsets(zoom_qty, prime_qty)
          gen_zoom_scales(zoom_qty).map do |o|
            gen_prime_offsets(prime_qty).map do |i|
              o * i
            end
          end
        end
      end
    end
  end
end
