require "./abstract"

module Ai4cr
  module Utils
    module IoData
      class TextFile < Ai4cr::Utils::IoData::Abstract
        BIT_32_INDEXES = (0..31).to_a

        def convert_raw_to_ios(raw)
          chars_as_bytes_of_bits = Array(Array(Float64)).new(raw.size)
          raw.each_char { |char| chars_as_bytes_of_bits << char_to_bits(char) }
          chars_as_bytes_of_bits
        end

        def char_to_bits(char)
          bytes = char.ord
          BIT_32_INDEXES.map { |i| bytes.bit(i) * 1.0 }
        end

        def bits_to_char(bits)
          bytes = BIT_32_INDEXES.map { |i| bits[i] * (2.0**i) }.sum.to_i
          bytes.chr.to_s
        end

        def bytes_to_chars(bytes)
          bytes.map { |bits| bits_to_char(bits) }
        end

        def convert_ios_to_raw(iod)
          bytes_to_chars(iod).join
        end
      end
    end
  end
end
