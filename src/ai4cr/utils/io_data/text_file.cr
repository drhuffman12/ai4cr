require "./abstract"

module Ai4cr
  module Utils
    module IoData
      class TextFile < Ai4cr::Utils::IoData::Abstract
        BIT_32_INDEXES   = (0..31).to_a
        UTF_MAX_AS_INT   = 0x10ffff
        UTF_MAX_AS_FLOAT = UTF_MAX_AS_INT.to_f

        def self.convert_raw_to_iod(raw)
          chars_as_bytes_of_bits = Array(Array(Float64)).new(raw.size)
          raw.each_char { |char| chars_as_bytes_of_bits << char_to_bits(char) }
          chars_as_bytes_of_bits
        end

        def self.char_to_bits(char)
          bytes = char.ord
          BIT_32_INDEXES.map { |i| bytes.bit(i) * 1.0 }
        end

        def self.bits_to_char(bits)
          # When converting back to a character, any 'fuzzy' bits must be forced to 0.0 or 1.0.
          indexes = bits.size < BIT_32_INDEXES.size ? (0..bits.size - 1).to_a : BIT_32_INDEXES
          bytes = indexes.sum do |i|
            bit = bits[i]
            # bit = if bit <= 0.0
            #         0.0
            #       elsif bit >= 1.0
            #         1.0
            #       else
            #         bit.round
            #       end

            bit = case
                  when bit.nan?
                    0.0
                  when bit <= 0.0
                    0.0
                  when bit >= 1.0
                    1.0
                  else
                    bit.round
                  end

            bit * (2.0**i)
          end
          bytes > UTF_MAX_AS_FLOAT ? UTF_MAX_AS_INT.chr.to_s : bytes.to_i.chr.to_s
        end

        def self.bytes_to_chars(bytes)
          bytes.map { |bits| bits_to_char(bits) }
        end

        def self.convert_iod_to_raw(iod)
          bytes_to_chars(iod).join("")
        end
      end
    end
  end
end
