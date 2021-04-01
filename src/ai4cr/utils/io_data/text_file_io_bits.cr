require "./abstract"

module Ai4cr
  module Utils
    module IoData
      class TextFileIodBits < Ai4cr::Utils::IoData::Abstract
        # TODO: Dig more into 'spec_bench/ai4cr/io_data/text_file_spec.cr' and adjust this file accordingly.
        BIT_32_INDEXES   = (0..31).to_a
        UTF_MAX_AS_FLOAT = Char::MAX_CODEPOINT.to_f

        def self.convert_raw_to_iod(raw, default_to_bit_size = BIT_32_INDEXES.size)
          chars_as_bytes_of_bits = Array(Array(Float64)).new(raw.size)
          raw.each_char { |char| chars_as_bytes_of_bits << char_to_bits(char, default_to_bit_size) }
          chars_as_bytes_of_bits
        end

        def self.char_to_bits(char, default_to_bit_size = BIT_32_INDEXES.size)
          bytes = char.ord
          bit_index_size(default_to_bit_size).map { |i| bytes.bit(i) * 1.0 }
        end

        def self.bit_index_size(default_to_bit_size = BIT_32_INDEXES.size)
          if default_to_bit_size <= 0
            BIT_32_INDEXES
          else
            (0..default_to_bit_size - 1)
          end
        end

        def self.bits_to_char(bits, default_to_bit_size = BIT_32_INDEXES.size)
          # When converting back to a character, any 'fuzzy' bits must be forced to 0.0 or 1.0.
          bidxs = bit_index_size(default_to_bit_size)
          indexes = bits.size < bidxs.size ? (0..bits.size - 1).to_a : bidxs
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
          bytes > UTF_MAX_AS_FLOAT ? Char::MAX_CODEPOINT.chr.to_s : bytes.to_i.chr.to_s
        end

        def self.bytes_to_chars(bytes, default_to_bit_size = BIT_32_INDEXES.size)
          bytes.map { |bits| bits_to_char(bits, default_to_bit_size) }
        end

        def self.convert_iod_to_raw(iod, default_to_bit_size = BIT_32_INDEXES.size)
          bytes_to_chars(iod, default_to_bit_size).join("")
        end
      end
    end
  end
end
